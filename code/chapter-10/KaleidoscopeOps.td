// KaleidoscopeOps.td

// Provides the TableGen definitions for MLIR dialects, types, and operations.
include "mlir/IR/OpBase.td"

def Kaleidoscope_Dialect : Dialect {
  // Operations and types from this dialect begin with `kaleidoscope`.
  let name = "kaleidoscope";

  // Generated C++ classes are placed in mlir::kaleidoscope.
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope variable semantics";

  // Use MLIR's generated parser and printer for our dialect's types. We will
  // see how their textual format is defined shortly.
  let useDefaultTypePrinterParser = 1;
}

// A common base for every operation in the Kaleidoscope dialect.
class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;

// TypeDef generates the mlir::kaleidoscope::VariableType C++ class. `Variable`
// is the generated C++ class stem; `var` is its textual mnemonic.
def Kaleidoscope_VariableType
    : TypeDef<Kaleidoscope_Dialect, "Variable"> {
  let mnemonic = "var";
  let summary = "a mutable Kaleidoscope variable";

  // The type carries no parameters, so nothing appears after its mnemonic. It
  // therefore prints simply as !kaleidoscope.var.
  let assemblyFormat = "";
}

// This record generates kaleidoscope::DeclareOp. The "var" argument to our
// base class supplies the operation mnemonic, producing `kaleidoscope.var`.
def Kaleidoscope_DeclareOp : Kaleidoscope_Op<"var", []> {
  let summary = "declare and initialize a mutable source variable";

  // `(ins ...)` describes operands and attributes accepted by the operation.
  // `$` names become generated C++ accessors such as getInitialValue() and
  // getName(). F64 is an SSA operand type; StrAttr and I64Attr are attributes
  // stored directly on the operation.
  let arguments = (ins F64:$initialValue, StrAttr:$name,
                       I64Attr:$argumentNumber);

  // `(outs ...)` describes SSA results. The result represents the source
  // variable, not the current floating-point value stored in that variable.
  let results = (outs Kaleidoscope_VariableType:$variable);

  // The assembly format controls parsing and printing. Backticks contain
  // literal punctuation, `$name` and `$initialValue` refer to fields above,
  // `attr-dict` prints any remaining attributes, and `type(...)` prints the
  // initial value's type.
  let assemblyFormat = "$name `=` $initialValue attr-dict `:` type($initialValue)";
}

def Kaleidoscope_ReadOp : Kaleidoscope_Op<"read", []> {
  let summary = "read a mutable source variable";

  // A read consumes a variable and produces its current f64 value.
  let arguments = (ins Kaleidoscope_VariableType:$variable);
  let results = (outs F64:$value);
  let assemblyFormat = "$variable attr-dict `:` type($value)";
}

def Kaleidoscope_AssignOp : Kaleidoscope_Op<"assign", []> {
  let summary = "assign a new value to a mutable source variable";

  // Assignment consumes the variable and replacement value. It has no
  // SSA result because BinaryExprAST returns the assigned value separately.
  let arguments = (ins Kaleidoscope_VariableType:$variable, F64:$value);
  let assemblyFormat = "$value `to` $variable attr-dict `:` type($value)";
}
