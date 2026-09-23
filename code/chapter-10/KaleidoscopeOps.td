// KaleidoscopeOps.td

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Kaleidoscope_Dialect : Dialect {
  let name = "kaleidoscope";
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope variable semantics";
  let useDefaultTypePrinterParser = 1;
}

class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;

def Kaleidoscope_VariableType
    : TypeDef<Kaleidoscope_Dialect, "Variable"> {
  let mnemonic = "var";
  let summary = "a mutable Kaleidoscope variable";
  let assemblyFormat = "";
}

def Kaleidoscope_DeclareOp : Kaleidoscope_Op<"var", []> {
  let summary = "declare and initialize a mutable source variable";
  let arguments = (ins F64:$initialValue, StrAttr:$name,
                       I64Attr:$argumentNumber);
  let results = (outs Kaleidoscope_VariableType:$variable);
  let assemblyFormat = "$name `=` $initialValue attr-dict `:` type($initialValue)";
}

def Kaleidoscope_ReadOp : Kaleidoscope_Op<"read", []> {
  let summary = "read a mutable source variable";
  let arguments = (ins Kaleidoscope_VariableType:$variable);
  let results = (outs F64:$value);
  let assemblyFormat = "$variable attr-dict `:` type($value)";
}

def Kaleidoscope_AssignOp : Kaleidoscope_Op<"assign", []> {
  let summary = "assign a new value to a mutable source variable";
  let arguments = (ins Kaleidoscope_VariableType:$variable, F64:$value);
  let assemblyFormat = "$value `to` $variable attr-dict `:` type($value)";
}
