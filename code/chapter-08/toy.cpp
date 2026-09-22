#include "../include/KaleidoscopeJIT.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_in = -10,

  // operators
  tok_binary = -11,
  tok_unary = -12,

  // var definition
  tok_var = -13
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

namespace {

/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;

  virtual Value codegen() = 0;
  virtual const std::string *getVariableName() const { return nullptr; }
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}

  Value codegen() override;
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}

  Value codegen() override;
  const std::string *getVariableName() const override { return &Name; }
};

/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value codegen() override;
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value codegen() override;
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

  Value codegen() override;
};

/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;

public:
  IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
            std::unique_ptr<ExprAST> Else)
      : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value codegen() override;
};

/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
             std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
             std::unique_ptr<ExprAST> Body)
      : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
        Step(std::move(Step)), Body(std::move(Body)) {}

  Value codegen() override;
};

/// VarExprAST - Expression class for var/in.
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;

public:
  VarExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
      : VarNames(std::move(VarNames)), Body(std::move(Body)) {}

  Value codegen() override;
};

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
  bool IsOperator;
  unsigned Precedence; // Precedence if a binary op.

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args,
               bool IsOperator = false, unsigned Prec = 0)
      : Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
        Precedence(Prec) {}

  func::FuncOp codegen();
  const std::string &getName() const { return Name; }
  const std::vector<std::string> &getArgs() const { return Args; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name.back();
  }

  unsigned getBinaryPrecedence() const { return Precedence; }
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}

  func::FuncOp codegen();
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");
  getNextToken(); // eat ).
  return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken(); // eat the if.

  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return LogError("expected then");
  getNextToken(); // eat the then.

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return LogError("expected else");
  getNextToken(); // eat the else.

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                     std::move(Else));
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken(); // eat the for.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken(); // eat '='.

  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken(); // eat the in.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                      std::move(Step), std::move(Body));
}

/// varexpr ::= 'var' identifier ('=' expression)?
///                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  if (CurTok != tok_identifier)
    return LogError("expected identifier after var");

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    std::unique_ptr<ExprAST> Init;
    if (CurTok == '=') {
      getNextToken(); // eat '='.
      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarNames.emplace_back(Name, std::move(Init));

    if (CurTok != ',')
      break;
    getNextToken(); // eat ','.
    if (CurTok != tok_identifier)
      return LogError("expected identifier list after var");
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' keyword after 'var'");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<VarExprAST>(std::move(VarNames), std::move(Body));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_var:
    return ParseVarExpr();
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary() {
  // If the current token is not an operator, it must be a primary expression.
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
    return ParsePrimary();

  // If this is a unary operator, read it.
  int Opc = CurTok;
  getNextToken();
  if (auto Operand = ParseUnary())
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}

/// binoprhs
///   ::= ('+' unary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = ParseUnary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS =
        std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  }
}

/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParseUnary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;
  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected unary operator");
    FnName = "unary";
    FnName += static_cast<char>(CurTok);
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected binary operator");
    FnName = "binary";
    FnName += static_cast<char>(CurTok);
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Invalid precedence: must be 1..100");
      BinaryPrecedence = static_cast<unsigned>(NumVal);
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  // Verify the right number of names for an operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Invalid number of operands for operator");

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames), Kind != 0,
                                        BinaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<MLIRContext> TheContext;
static OwningOpRef<ModuleOp> TheModule;
static std::unique_ptr<OpBuilder> TheBuilder;
static std::unique_ptr<PassManager> ThePM;
static std::map<std::string, Value> NamedValues;
static std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static llvm::ExitOnError ExitOnErr;
static llvm::cl::opt<bool> DumpMLIR(
    "dump-mlir", llvm::cl::desc("Print generated MLIR"),
    llvm::cl::init(false));
static llvm::cl::opt<bool>
    EmitObject("emit-object",
               llvm::cl::desc("Compile all input to output.o instead of using "
                              "the JIT"),
               llvm::cl::init(false));
static llvm::cl::opt<bool>
    DumpLLVMIR("dump-llvm-ir",
             llvm::cl::desc("Print LLVM IR before emitting the object file"),
             llvm::cl::init(false));

static Location getLocation() { return TheBuilder->getUnknownLoc(); }

Value LogErrorV(const char *Str) {
  LogError(Str);
  return {};
}

func::FuncOp getFunction(const std::string &Name) {
  // First, see if the function has already been added to the current module.
  if (auto Function = TheModule->lookupSymbol<func::FuncOp>(Name))
    return Function;

  // If not, codegen the declaration from an existing prototype.
  auto It = FunctionProtos.find(Name);
  if (It != FunctionProtos.end()) {
    auto Function = It->second->codegen();
    Function.setPrivate();
    return Function;
  }

  return {};
}

static func::FuncOp getCurrentFunction() {
  Operation *Parent = TheBuilder->getInsertionBlock()->getParentOp();
  if (auto Function = dyn_cast<func::FuncOp>(Parent))
    return Function;
  return Parent->getParentOfType<func::FuncOp>();
}

/// CreateEntryBlockAlloca - Create mutable storage in the function entry block.
static Value CreateEntryBlockAlloca() {
  func::FuncOp Function = getCurrentFunction();
  OpBuilder::InsertionGuard Guard(*TheBuilder);
  TheBuilder->setInsertionPointToStart(&Function.front());
  auto VariableType = MemRefType::get({}, TheBuilder->getF64Type());
  return TheBuilder->create<memref::AllocaOp>(getLocation(), VariableType);
}

Value NumberExprAST::codegen() {
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(Val));
}

Value VariableExprAST::codegen() {
  // Look this variable up in the function.
  auto It = NamedValues.find(Name);
  if (It == NamedValues.end())
    return LogErrorV("Unknown variable name");

  return TheBuilder->create<memref::LoadOp>(getLocation(), It->second,
                                             ValueRange{});
}

Value UnaryExprAST::codegen() {
  Value OperandV = Operand->codegen();
  if (!OperandV)
    return {};

  auto Operator = getFunction(std::string("unary") + Opcode);
  if (!Operator)
    return LogErrorV("Unknown unary operator");

  return TheBuilder->create<func::CallOp>(getLocation(), Operator, OperandV)
      .getResult(0);
}

Value BinaryExprAST::codegen() {
  // Assignment stores into the variable's mutable memref slot.
  if (Op == '=') {
    const std::string *Name = LHS->getVariableName();
    if (!Name)
      return LogErrorV("destination of '=' must be a variable");

    Value AssignedValue = RHS->codegen();
    if (!AssignedValue)
      return {};

    auto It = NamedValues.find(*Name);
    if (It == NamedValues.end())
      return LogErrorV("Unknown variable name");

    TheBuilder->create<memref::StoreOp>(getLocation(), AssignedValue,
                                        It->second, ValueRange{});
    return AssignedValue;
  }

  Value L = LHS->codegen();
  Value R = RHS->codegen();
  if (!L || !R)
    return {};

  switch (Op) {
  case '+':
    return TheBuilder->create<arith::AddFOp>(getLocation(), L, R);
  case '-':
    return TheBuilder->create<arith::SubFOp>(getLocation(), L, R);
  case '*':
    return TheBuilder->create<arith::MulFOp>(getLocation(), L, R);
  case '<': {
    Value Comparison = TheBuilder->create<arith::CmpFOp>(
        getLocation(), arith::CmpFPredicate::ULT, L, R);
    // Convert bool 0/1 to double 0.0 or 1.0.
    return TheBuilder->create<arith::UIToFPOp>(
        getLocation(), TheBuilder->getF64Type(), Comparison);
  }
  default:
    break;
  }

  // If it wasn't a builtin binary operator, it must be a user-defined one.
  auto Operator = getFunction(std::string("binary") + Op);
  if (!Operator)
    return LogErrorV("Unknown binary operator");

  Value Operands[] = {L, R};
  return TheBuilder->create<func::CallOp>(getLocation(), Operator, Operands)
      .getResult(0);
}

Value CallExprAST::codegen() {
  // Look up the name in the global module table.
  auto CalleeF = getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF.getNumArguments() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value> ArgsV;
  for (auto &Arg : Args) {
    ArgsV.push_back(Arg->codegen());
    if (!ArgsV.back())
      return {};
  }

  return TheBuilder->create<func::CallOp>(getLocation(), CalleeF, ArgsV)
      .getResult(0);
}

Value IfExprAST::codegen() {
  Value CondV = Cond->codegen();
  if (!CondV)
    return {};

  // Convert the condition to a boolean by comparing it with 0.0.
  Value Zero = TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(0.0));
  CondV = TheBuilder->create<arith::CmpFOp>(
      getLocation(), arith::CmpFPredicate::ONE, CondV, Zero);

  bool CodegenFailed = false;
  auto IfOp = TheBuilder->create<scf::IfOp>(
      getLocation(), CondV,
      [&](OpBuilder &Builder, Location Loc) {
        Value ThenV = Then->codegen();
        if (!ThenV) {
          CodegenFailed = true;
          ThenV = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }
        Builder.create<scf::YieldOp>(Loc, ThenV);
      },
      [&](OpBuilder &Builder, Location Loc) {
        Value ElseV = Else->codegen();
        if (!ElseV) {
          CodegenFailed = true;
          ElseV = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }
        Builder.create<scf::YieldOp>(Loc, ElseV);
      });

  if (CodegenFailed)
    return {};
  return IfOp.getResult(0);
}

Value ForExprAST::codegen() {
  // Emit the start value before putting the loop variable in scope.
  Value StartVal = Start->codegen();
  if (!StartVal)
    return {};

  Value Variable = CreateEntryBlockAlloca();
  TheBuilder->create<memref::StoreOp>(getLocation(), StartVal, Variable,
                                      ValueRange{});

  auto OldValue = NamedValues.find(VarName);
  bool HadOldValue = OldValue != NamedValues.end();
  Value SavedValue = HadOldValue ? OldValue->second : Value();
  NamedValues[VarName] = Variable;
  bool CodegenFailed = false;

  // Test the condition before each iteration, then emit the body and step.
  TheBuilder->create<scf::WhileOp>(
      getLocation(), TypeRange{}, ValueRange{},
      [&](OpBuilder &Builder, Location Loc, ValueRange) {
        Value EndCond = End->codegen();
        if (!EndCond) {
          CodegenFailed = true;
          EndCond = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }

        Value Zero = Builder.create<arith::ConstantOp>(
            Loc, Builder.getF64FloatAttr(0.0));
        EndCond = Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::ONE,
                                                EndCond, Zero);
        Builder.create<scf::ConditionOp>(Loc, EndCond, ValueRange{});
      },
      [&](OpBuilder &Builder, Location Loc, ValueRange) {
        if (!Body->codegen())
          CodegenFailed = true;

        Value StepVal;
        if (Step)
          StepVal = Step->codegen();
        else
          StepVal = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(1.0));
        if (!StepVal) {
          CodegenFailed = true;
          StepVal = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(1.0));
        }

        // Reload after the body and step in case either mutated the variable.
        Value Current =
            Builder.create<memref::LoadOp>(Loc, Variable, ValueRange{});
        Value NextVar = Builder.create<arith::AddFOp>(Loc, Current, StepVal);
        Builder.create<memref::StoreOp>(Loc, NextVar, Variable, ValueRange{});
        Builder.create<scf::YieldOp>(Loc);
      });

  // Restore any variable shadowed by the loop induction variable.
  if (HadOldValue)
    NamedValues[VarName] = SavedValue;
  else
    NamedValues.erase(VarName);

  if (CodegenFailed)
    return {};

  // A for expression always returns 0.0.
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(0.0));
}

Value VarExprAST::codegen() {
  std::vector<std::pair<std::string, std::optional<Value>>> OldBindings;

  auto RestoreBindings = [&]() {
    for (auto It = OldBindings.rbegin(); It != OldBindings.rend(); ++It) {
      if (It->second)
        NamedValues[It->first] = *It->second;
      else
        NamedValues.erase(It->first);
    }
  };

  for (auto &Variable : VarNames) {
    const std::string &Name = Variable.first;

    // Generate the initializer before introducing the new binding.
    Value InitialValue;
    if (Variable.second)
      InitialValue = Variable.second->codegen();
    else
      InitialValue = TheBuilder->create<arith::ConstantOp>(
          getLocation(), TheBuilder->getF64FloatAttr(0.0));
    if (!InitialValue) {
      RestoreBindings();
      return {};
    }

    Value Storage = CreateEntryBlockAlloca();
    TheBuilder->create<memref::StoreOp>(getLocation(), InitialValue, Storage,
                                        ValueRange{});

    auto Old = NamedValues.find(Name);
    OldBindings.emplace_back(
        Name, Old == NamedValues.end() ? std::optional<Value>()
                                      : std::optional<Value>(Old->second));
    NamedValues[Name] = Storage;
  }

  Value BodyValue = Body->codegen();
  RestoreBindings();
  return BodyValue;
}

func::FuncOp PrototypeAST::codegen() {
  // Make the function type: double(double, double), etc.
  std::vector<Type> Doubles(Args.size(), TheBuilder->getF64Type());
  auto FunctionType =
      TheBuilder->getFunctionType(Doubles, {TheBuilder->getF64Type()});

  auto Function = func::FuncOp::create(getLocation(), Name, FunctionType);
  TheModule->push_back(Function);
  return Function;
}

func::FuncOp FunctionAST::codegen() {
  // Save the prototype so declarations can be emitted in later modules.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  auto TheFunction = getFunction(P.getName());

  if (!TheFunction)
    return {};

  if (!TheFunction.isDeclaration()) {
    LogError("Function cannot be redefined.");
    return {};
  }

  // A definition is visible outside the module, even if an earlier extern
  // declaration created the function with private symbol visibility.
  TheFunction.setPublic();

  // If this is a binary operator, install its precedence.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  Block *EntryBlock = TheFunction.addEntryBlock();
  TheBuilder->setInsertionPointToStart(EntryBlock);

  // Give each function argument a mutable storage slot.
  NamedValues.clear();
  unsigned Index = 0;
  for (BlockArgument Argument : TheFunction.getArguments()) {
    Value Storage = CreateEntryBlockAlloca();
    TheBuilder->create<memref::StoreOp>(getLocation(), Argument, Storage,
                                        ValueRange{});
    NamedValues[P.getArgs()[Index++]] = Storage;
  }

  if (Value RetVal = Body->codegen()) {
    // Finish off the function.
    TheBuilder->create<func::ReturnOp>(getLocation(), RetVal);

    // Validate the generated code, checking for consistency.
    if (succeeded(verify(TheFunction))) {
      // Run the optimizer on the module.
      if (failed(ThePM->run(*TheModule))) {
        LogError("Could not optimize function.");
        TheFunction.erase();
        if (P.isBinaryOp())
          BinopPrecedence.erase(P.getOperatorName());
        return {};
      }
      return TheFunction;
    }
  }

  // Error reading body, remove function.
  TheFunction.erase();
  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  return {};
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModuleAndManagers() {
  // Destroy objects that refer to the old context before replacing it.
  ThePM.reset();
  TheBuilder.reset();
  TheModule = OwningOpRef<ModuleOp>();
  TheContext.reset();

  // Open a new context and module.
  TheContext = std::make_unique<MLIRContext>();
  TheContext->loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
                          func::FuncDialect, memref::MemRefDialect,
                          scf::SCFDialect>();
  TheModule = ModuleOp::create(UnknownLoc::get(TheContext.get()));

  // Create a new builder for the module.
  TheBuilder = std::make_unique<OpBuilder>(TheContext.get());

  // Create a pass manager and add a couple of simple optimizations.
  ThePM = std::make_unique<PassManager>(TheContext.get());
  ThePM->addNestedPass<func::FuncOp>(createCanonicalizerPass());
  ThePM->addNestedPass<func::FuncOp>(createCSEPass());
}

struct LoweredModule {
  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::Module> Module;
};

static llvm::Expected<LoweredModule>
lowerToLLVM(const llvm::DataLayout &DataLayout) {
  // Lower the high-level MLIR operations to the LLVM dialect.
  PassManager LoweringPM(TheContext.get());
  LoweringPM.addPass(createSCFToControlFlowPass());
  LoweringPM.addPass(createConvertFuncToLLVMPass());
  LoweringPM.addPass(createArithToLLVMConversionPass());
  LoweringPM.addPass(createFinalizeMemRefToLLVMConversionPass());
  LoweringPM.addPass(createConvertControlFlowToLLVMPass());

  // Clean up any temporary casts introduced by dialect conversion.
  LoweringPM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(LoweringPM.run(*TheModule)))
    return llvm::make_error<llvm::StringError>(
        "could not lower module to the LLVM dialect",
        llvm::inconvertibleErrorCode());

  // Register the translations from MLIR's LLVM dialect to LLVM IR.
  registerBuiltinDialectTranslation(*TheContext);
  registerLLVMDialectTranslation(*TheContext);

  // Translate the lowered MLIR module into the LLVM IR module consumed by the
  // target's object-file emitter.
  auto LLVMContext = std::make_unique<llvm::LLVMContext>();
  auto LLVMModule = translateModuleToLLVMIR(*TheModule, *LLVMContext);
  if (!LLVMModule)
    return llvm::make_error<llvm::StringError>(
        "could not translate the LLVM dialect to LLVM IR",
        llvm::inconvertibleErrorCode());

  LLVMModule->setDataLayout(DataLayout);

  if (DumpLLVMIR) {
    LLVMModule->print(llvm::errs(), nullptr);
    llvm::errs() << '\n';
  }

  return LoweredModule{std::move(LLVMContext), std::move(LLVMModule)};
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read function definition:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      if (!EmitObject) {
        auto Lowered = ExitOnErr(lowerToLLVM(TheJIT->getDataLayout()));
        ExitOnErr(TheJIT->addModule(llvm::orc::ThreadSafeModule(
            std::move(Lowered.Module), std::move(Lowered.Context))));
        InitializeModuleAndManagers();
      }

    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto FnIR = ProtoAST->codegen()) {
      FnIR.setPrivate();
      if (DumpMLIR) {
        fprintf(stderr, "Read extern:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression with the JIT, or retain it when compiling.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read top-level expression:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      if (!EmitObject) {
        auto RT = TheJIT->getMainJITDylib().createResourceTracker();
        auto Lowered = ExitOnErr(lowerToLLVM(TheJIT->getDataLayout()));
        ExitOnErr(TheJIT->addModule(
            llvm::orc::ThreadSafeModule(std::move(Lowered.Module),
                                        std::move(Lowered.Context)),
            RT));
        InitializeModuleAndManagers();

        auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
        double (*FP)() = ExprSymbol.getAddress().toPtr<double (*)()>();
        fprintf(stderr, "Evaluated to %f\n", FP());

        ExitOnErr(RT->remove());
      }
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a double, prints it as "%f\n", and returns 0.
extern "C" DLLEXPORT double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      fprintf(stderr, "ready> ");
      getNextToken();
      continue;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
    if (CurTok != tok_eof && CurTok != ';')
      fprintf(stderr, "ready> ");
  }
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Kaleidoscope object file compiler\n");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  if (!EmitObject)
    TheJIT = ExitOnErr(llvm::orc::KaleidoscopeJIT::Create());

  // JIT mode replaces this module as definitions are submitted. Object mode
  // retains it until the entire input has been parsed.
  InitializeModuleAndManagers();

  // Run the main "interpreter loop" now.
  MainLoop();

  if (!EmitObject)
    return 0;

  // Select the host target and configure its object-file emitter.
  auto TargetTriple = llvm::sys::getDefaultTargetTriple();
  std::string Error;
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    llvm::errs() << Error << '\n';
    return 1;
  }

  llvm::TargetOptions Options;
  std::unique_ptr<llvm::TargetMachine> TargetMachine(
      Target->createTargetMachine(llvm::Triple(TargetTriple), "generic", "",
                                  Options, llvm::Reloc::PIC_));
  if (!TargetMachine) {
    llvm::errs() << "Could not create the target machine\n";
    return 1;
  }

  // Lower the complete MLIR module once, then attach the target information
  // required to produce a native object file.
  auto Lowered = ExitOnErr(lowerToLLVM(TargetMachine->createDataLayout()));
  Lowered.Module->setTargetTriple(llvm::Triple(TargetTriple));

  const char *Filename = "output.o";
  std::error_code EC;
  llvm::raw_fd_ostream Dest(Filename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << '\n';
    return 1;
  }

  llvm::legacy::PassManager EmitPM;
  if (TargetMachine->addPassesToEmitFile(
          EmitPM, Dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit an object file\n";
    return 1;
  }

  EmitPM.run(*Lowered.Module);
  Dest.flush();
  llvm::outs() << "Wrote " << Filename << '\n';

  return 0;
}
