// KaleidoscopeOps.td
//
// Defines the Kaleidoscope dialect: a small set of operations that preserve
// source-level variable semantics (name, location, argument number) in the IR
// until lowering is ready to turn them into stack storage and debug info.

// Provides the TableGen definitions for MLIR dialects, types, and operations.
include "mlir/IR/OpBase.td"

def Kaleidoscope_Dialect : Dialect {
  // The prefix for every operation and type in this dialect. An operation
  // with the mnemonic `var` will therefore print as `kaleidoscope.var`.
  let name = "kaleidoscope";

  // Where the generated C++ classes live. Our classes end up in the
  // `mlir::kaleidoscope` namespace.
  let cppNamespace = "::mlir::kaleidoscope";

  let summary = "Operations that preserve Kaleidoscope variable semantics";

  // Ask MLIR to generate the parser and printer for our types based on the
  // assembly format each type declares. We only have one type, and its format
  // is defined below.
  let useDefaultTypePrinterParser = 1;
}

// A common base for every operation in this dialect. Each concrete operation
// supplies its own mnemonic and, optionally, a list of traits describing its
// behavior.
class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;

// A type representing a mutable source variable.
//
// TypeDef generates the C++ class `mlir::kaleidoscope::VariableType` from the
// class stem "Variable", and the mnemonic "var" gives it the textual spelling
// `!kaleidoscope.var`.
//
// The type deliberately says nothing about how the variable is stored. That
// is a lowering decision, not a property of the source language.
def Kaleidoscope_VariableType
    : TypeDef<Kaleidoscope_Dialect, "Variable"> {
  let mnemonic = "var";
  let summary = "a mutable Kaleidoscope variable";

  // An empty assembly format means the type has no parameters to print after
  // its mnemonic, so it always appears as the bare `!kaleidoscope.var`.
  let assemblyFormat = "";
}

// Declares and initializes a mutable source variable.
//
// This generates the C++ class `kaleidoscope::DeclareOp`, printed in IR as
// `kaleidoscope.var`. It is the operation that carries the source name and
// argument number that would otherwise be lost the moment a variable became
// an anonymous allocation.
def Kaleidoscope_DeclareOp : Kaleidoscope_Op<"var", []> {
  let summary = "declare and initialize a mutable source variable";

  // `(ins ...)` lists everything the operation takes in. The `$` names
  // generate C++ accessors (getInitialValue, getName, getArgumentNumber).
  //
  //   - F64:$initialValue   is an SSA operand, constrained to f64
  //   - StrAttr:$name       is an attribute holding the source name
  //   - I64Attr:$argumentNumber is an attribute; 0 means a local variable,
  //                         and 1+ means a function parameter (DWARF numbers
  //                         parameters starting at 1)
  let arguments = (ins F64:$initialValue, StrAttr:$name,
                       I64Attr:$argumentNumber);

  // The SSA result represents the *variable itself*, not the floating-point
  // value currently stored in it. Subsequent read and assign operations use
  // this value to refer to the variable.
  let results = (outs Kaleidoscope_VariableType:$variable);

  // The textual syntax. Backticks contain literal punctuation, `$name` and
  // `$initialValue` refer to the fields above, `attr-dict` prints any
  // attributes not already consumed by the format, and `type(...)` prints the
  // type of the named operand.
  //
  // This produces, for example:
  //   %0 = kaleidoscope.var "x" = %arg0 {argumentNumber = 1 : i64} : f64
  let assemblyFormat = "$name `=` $initialValue attr-dict `:` type($initialValue)";
}

// Reads the current value of a mutable source variable.
//
// Not marked `Pure`: two reads of the same variable are not necessarily equal,
// since an assignment may occur between them. Marking it pure would let CSE
// incorrectly collapse the two reads into one.
def Kaleidoscope_ReadOp : Kaleidoscope_Op<"read", []> {
  let summary = "read a mutable source variable";

  // Consumes the variable; produces the f64 value currently stored in it.
  let arguments = (ins Kaleidoscope_VariableType:$variable);
  let results = (outs F64:$value);
  let assemblyFormat = "$variable attr-dict `:` type($value)";
}

// Assigns a new value to a mutable source variable.
//
// The operation itself produces no SSA result. The AST layer returns the
// assigned value separately so that an assignment expression can be used as a
// subexpression (as in `(y = y + 1) * y`).
def Kaleidoscope_AssignOp : Kaleidoscope_Op<"assign", []> {
  let summary = "assign a new value to a mutable source variable";

  // Consumes the variable and the replacement value.
  let arguments = (ins Kaleidoscope_VariableType:$variable, F64:$value);
  let assemblyFormat = "$value `to` $variable attr-dict `:` type($value)";
}