include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Kaleidoscope_Dialect : Dialect {
  let name = "kaleidoscope";
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope language semantics";
  let hasConstantMaterializer = 1;
}

class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;

def Kaleidoscope_BinaryOp : Kaleidoscope_Op<"binary", [
    SameOperandsAndResultType
  ]> {
  let summary = "a built-in or user-defined Kaleidoscope binary operator";
  let arguments = (ins F64:$lhs, F64:$rhs, StrAttr:$operatorName);
  let results = (outs F64:$result);

  let builders = [
    OpBuilder<(ins "StringRef":$operatorName, "Value":$lhs,
                   "Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs,
            $_builder.getStringAttr(operatorName));
    }]>
  ];

  let assemblyFormat = "$operatorName $lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}
