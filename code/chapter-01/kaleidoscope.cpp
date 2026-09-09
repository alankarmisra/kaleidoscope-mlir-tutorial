#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

//===----------------------------------------===//
// Command line
//===----------------------------------------===//
static llvm::cl::OptionCategory kaleidoscopeOptions("Kaleidoscope options");

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input.ks>"),
                                            llvm::cl::Required,
                                            llvm::cl::cat(kaleidoscopeOptions));

static llvm::cl::opt<bool> dumpTokens("dump-tokens",
                                      llvm::cl::desc("Print lexer tokens"),
                                      llvm::cl::init(false),
                                      llvm::cl::cat(kaleidoscopeOptions));

//===----------------------------------------===//
// Lexer
//===----------------------------------------===//

enum class TokenKind {
  invalid,
  eof,
  defKeyword,
  returnKeyword,
  identifier,
  integer,
  leftParenthesis,
  rightParenthesis,
  arrow,
  colon,
  newline,
  indent,
  dedent,
};

static llvm::StringRef tokenKindName(TokenKind kind) {
  switch (kind) {
  case TokenKind::invalid:
    return "invalid";
  case TokenKind::eof:
    return "eof";
  case TokenKind::defKeyword:
    return "def";
  case TokenKind::returnKeyword:
    return "return";
  case TokenKind::identifier:
    return "identifier";
  case TokenKind::integer:
    return "integer";
  case TokenKind::leftParenthesis:
    return "left parenthesis";
  case TokenKind::rightParenthesis:
    return "right parenthesis";
  case TokenKind::arrow:
    return "arrow";
  case TokenKind::colon:
    return "colon";
  case TokenKind::newline:
    return "newline";
  case TokenKind::indent:
    return "indent";
  case TokenKind::dedent:
    return "dedent";
  }

  llvm_unreachable("unknown token kind");
}

llvm::StringMap<TokenKind> Keywords = {{"def", TokenKind::defKeyword},
                                       {"return", TokenKind::returnKeyword}};

struct Token {
  TokenKind kind;
  llvm::StringRef spelling;
  unsigned line;
  unsigned column;
};

class Lexer {
public:
  explicit Lexer(llvm::StringRef source) : source(source) {}

  Token next() {
    if (pendingDedents > 0) {
      --pendingDedents;
      return makeEmptyToken(TokenKind::dedent);
    }

    if (atBeginningOfLine) {
      std::optional<Token> indentationToken = lexIndentation();
      if (indentationToken)
        return *indentationToken;
    }

    while (peek() == ' ')
      advance();

    if (atEnd()) {
      if (indentation.size() > 1) {
        indentation.pop_back();
        return makeEmptyToken(TokenKind::dedent);
      }

      return makeEmptyToken(TokenKind::eof);
    }

    unsigned tokenLine = line;
    unsigned tokenColumn = column;
    std::size_t start = offset;

    if (peek() == '\n' || peek() == '\r') {
      consumeNewLine();
      return makeToken(TokenKind::newline, start, tokenLine, tokenColumn);
    }

    if (llvm::isAlpha(peek()) || peek() == '_')
      return lexIdentifier();

    if (llvm::isDigit(peek()))
      return lexInteger();

    TokenKind kind = TokenKind::invalid;
    switch (peek()) {
    case '(':
      advance();
      kind = TokenKind::leftParenthesis;
      break;
    case ')':
      advance();
      kind = TokenKind::rightParenthesis;
      break;
    case ':':
      advance();
      kind = TokenKind::colon;
      break;
    case '-':
      advance();
      if (peek() == '>') {
        advance();
        kind = TokenKind::arrow;
      } else {
        kind = TokenKind::invalid;
      }
      break;
    default:
      advance();
      break;
    }
    return makeToken(kind, start, tokenLine, tokenColumn);
  }

private:
  llvm::StringRef source;
  std::size_t offset = 0;
  unsigned line = 1;
  unsigned column = 1;
  bool atBeginningOfLine = true;
  std::vector<unsigned> indentation{0};
  unsigned pendingDedents = 0;

  bool atEnd() const { return offset == source.size(); }

  char peek() const {
    if (atEnd())
      return '\0';

    return source[offset];
  }

  char advance() {
    char character = source[offset++];
    ++column;
    return character;
  }

  void consumeNewLine() {
    if (peek() == '\r')
      advance();
    if (peek() == '\n')
      advance();

    ++line;
    column = 1;
    atBeginningOfLine = true;
  }

  Token makeToken(TokenKind kind, std::size_t start, unsigned tokenLine,
                  unsigned tokenColumn) const {
    return {kind, source.slice(start, offset), tokenLine, tokenColumn};
  }

  Token makeEmptyToken(TokenKind kind) const {
    return {kind, llvm::StringRef(), line, column};
  }

  Token lexIdentifier() {
    std::size_t start = offset;
    unsigned tokenLine = line;
    unsigned tokenColumn = column;

    while (llvm::isAlnum(peek()) || peek() == '_')
      advance();

    llvm::StringRef spelling = source.slice(start, offset);
    TokenKind kind = TokenKind::identifier;
    auto keywordIterator = Keywords.find(spelling);
    if (keywordIterator != Keywords.end())
      kind = keywordIterator->second;
    return {kind, spelling, tokenLine, tokenColumn};
  }

  Token lexInteger() {
    std::size_t start = offset;
    unsigned tokenLine = line;
    unsigned tokenColumn = column;

    while (llvm::isDigit(peek()))
      advance();

    return makeToken(TokenKind::integer, start, tokenLine, tokenColumn);
  }

  std::optional<Token> lexIndentation() {
    std::size_t start = offset;
    unsigned tokenLine = line;
    unsigned tokenColumn = column;
    unsigned spaces = 0;

    // TODO: so you don't consider tabs?
    while (peek() == ' ') {
      advance();
      ++spaces;
    }

    // Empty lines do not open or close indentation levels;
    // TODO: Why won't this look for EOF?
    if (peek() == '\n' || peek() == '\r' || atEnd()) {
      atBeginningOfLine = false;
      return std::nullopt;
    }

    atBeginningOfLine = false;

    // indent
    // TODO: So your indentations are just 1 space?
    if (spaces > indentation.back()) {
      indentation.push_back(spaces);
      return makeToken(TokenKind::indent, start, tokenLine, tokenColumn);
    }

    // dedent
    if (spaces < indentation.back()) {
      // TODO: So your indentations are just 1 space?
      while (spaces < indentation.back()) {
        indentation.pop_back();
        ++pendingDedents;
      }

      // A line must return to an indentation level seen previously
      if (spaces != indentation.back()) {
        return makeToken(TokenKind::invalid, start, tokenLine, tokenColumn);
      }

      --pendingDedents;
      return makeEmptyToken(TokenKind::dedent);
    }

    return std::nullopt;
  }
};

//===----------------------------------------===//
// Abstract syntax tree
//===----------------------------------------===//

struct SourceLocation {
  unsigned line;
  unsigned column;
};

int main(int argc, char **argv) {
  llvm::cl::HideUnrelatedOptions(kaleidoscopeOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Kaleidoscope MLIR compiler\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> source =
      llvm::MemoryBuffer::getFile(inputFile.getValue());
  if (!source) {
    llvm::errs() << "error: could not read file: " << inputFile.getValue()
                 << ": " << source.getError().message() << '\n';
    return 1;
  }

  if (dumpTokens) {
    Lexer lexer((*source)->getBuffer());
    for (;;) {
      Token token = lexer.next();

      llvm::errs() << tokenKindName(token.kind) << " at " << token.line << ":"
                   << token.column;

      if (!token.spelling.empty() && token.kind != TokenKind::newline)
        llvm::errs() << " [" << token.spelling << "]";

      llvm::errs() << '\n';

      if (token.kind == TokenKind::eof)
        break;
    }

    return 0;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);

  mlir::Location location = builder.getUnknownLoc();
  mlir::ModuleOp module = mlir::ModuleOp::create(location);

  mlir::FunctionType functionType =
      builder.getFunctionType({}, {builder.getI32Type()});

  mlir::func::FuncOp mainFunction =
      mlir::func::FuncOp::create(location, "main", functionType);

  mlir::Block *entryBlock = mainFunction.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::arith::ConstantIntOp result =
      builder.create<mlir::arith::ConstantIntOp>(location, 42, 32);

  builder.create<mlir::func::ReturnOp>(location, result.getResult());

  module.push_back(mainFunction);
  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Generated invalid MLIR\n";
    return 1;
  }
  module.print(llvm::outs());
  llvm::outs() << '\n';

  return 0;
}
