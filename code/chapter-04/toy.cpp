#include "../include/KaleidoscopeJIT.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
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
  tok_number = -5
};

static std::string IdentifierStr; // Filled in for identifiers and keywords
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

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}

  func::FuncOp codegen();
  const std::string &getName() const { return Name; }
  const std::vector<std::string> &getArgs() const { return Args; }
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

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
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
  }
}

/// binoprhs
///   ::= ('+' primary)*
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

    // Parse the primary expression after the binary operator.
    auto RHS = ParsePrimary();
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
///   ::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != tok_identifier)
    return LogErrorP("Expected function name in prototype");

  std::string FnName = IdentifierStr;
  getNextToken();

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
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
static llvm::cl::opt<bool> DumpLLVMIR(
    "dump-llvm-ir", llvm::cl::desc("Print translated LLVM IR"),
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

Value NumberExprAST::codegen() {
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(Val));
}

Value VariableExprAST::codegen() {
  // Look this variable up in the function.
  auto It = NamedValues.find(Name);
  if (It == NamedValues.end())
    return LogErrorV("Unknown variable name");
  return It->second;
}

Value BinaryExprAST::codegen() {
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
    return LogErrorV("invalid binary operator");
  }
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

  // Create a new basic block to start insertion into.
  Block *EntryBlock = TheFunction.addEntryBlock();
  TheBuilder->setInsertionPointToStart(EntryBlock);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  unsigned Index = 0;
  for (BlockArgument Argument : TheFunction.getArguments())
    NamedValues[P.getArgs()[Index++]] = Argument;

  if (Value RetVal = Body->codegen()) {
    // Finish off the function.
    TheBuilder->create<func::ReturnOp>(getLocation(), RetVal);

    // Validate the generated code, checking for consistency.
    if (succeeded(verify(TheFunction))) {
      // Run the optimizer on the module.
      if (failed(ThePM->run(*TheModule))) {
        LogError("Could not optimize function.");
        TheFunction.erase();
        return {};
      }
      return TheFunction;
    }
  }

  // Error reading body, remove function.
  TheFunction.erase();
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
  TheContext->loadDialect<arith::ArithDialect, func::FuncDialect>();
  TheModule = ModuleOp::create(UnknownLoc::get(TheContext.get()));

  // Create a new builder for the module.
  TheBuilder = std::make_unique<OpBuilder>(TheContext.get());

  // Create a pass manager and add a couple of simple optimizations.
  ThePM = std::make_unique<PassManager>(TheContext.get());
  ThePM->addNestedPass<func::FuncOp>(createCanonicalizerPass());
  ThePM->addNestedPass<func::FuncOp>(createCSEPass());
}

static llvm::Expected<llvm::orc::ThreadSafeModule> lowerToLLVM() {
  // Lower the high-level MLIR operations to the LLVM dialect.
  PassManager LoweringPM(TheContext.get());
  LoweringPM.addPass(createConvertFuncToLLVMPass());
  LoweringPM.addPass(createArithToLLVMConversionPass());

  // Clean up any temporary casts introduced by dialect conversion.
  LoweringPM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(LoweringPM.run(*TheModule)))
    return llvm::make_error<llvm::StringError>(
        "could not lower module to the LLVM dialect",
        llvm::inconvertibleErrorCode());

  // Register the translations from MLIR's LLVM dialect to LLVM IR.
  registerBuiltinDialectTranslation(*TheContext);
  registerLLVMDialectTranslation(*TheContext);

  // Translate the lowered MLIR module into an LLVM IR module. The LLVM
  // context is kept with the module because the JIT may compile it later.
  auto LLVMContext = std::make_unique<llvm::LLVMContext>();
  auto LLVMModule = translateModuleToLLVMIR(*TheModule, *LLVMContext);
  if (!LLVMModule)
    return llvm::make_error<llvm::StringError>(
        "could not translate the LLVM dialect to LLVM IR",
        llvm::inconvertibleErrorCode());

  // Match the module's data layout to the target selected by the JIT.
  LLVMModule->setDataLayout(TheJIT->getDataLayout());

  if (DumpLLVMIR) {
    LLVMModule->print(llvm::errs(), nullptr);
    llvm::errs() << '\n';
  }

  // ThreadSafeModule transfers ownership of both objects to the ORC JIT.
  return llvm::orc::ThreadSafeModule(std::move(LLVMModule),
                                     std::move(LLVMContext));
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read function definition:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      ExitOnErr(TheJIT->addModule(ExitOnErr(lowerToLLVM())));
      InitializeModuleAndManagers();
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
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read top-level expression:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      auto RT = TheJIT->getMainJITDylib().createResourceTracker();
      ExitOnErr(TheJIT->addModule(ExitOnErr(lowerToLLVM()), RT));
      InitializeModuleAndManagers();

      auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
      double (*FP)() = ExprSymbol.getAddress().toPtr<double (*)()>();
      fprintf(stderr, "Evaluated to %f\n", FP());

      ExitOnErr(RT->remove());
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
  llvm::cl::ParseCommandLineOptions(argc, argv, "Kaleidoscope JIT\n");
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(llvm::orc::KaleidoscopeJIT::Create());

  // Make the first module, which holds newly generated code.
  InitializeModuleAndManagers();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
