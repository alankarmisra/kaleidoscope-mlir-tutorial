#include "KaleidoscopeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mlir::kaleidoscope;

#include "KaleidoscopeDialect.cpp.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.cpp.inc"

void KaleidoscopeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KaleidoscopeOps.cpp.inc"
      >();
}

Operation *KaleidoscopeDialect::materializeConstant(OpBuilder &Builder,
                                                    Attribute Value, Type Type,
                                                    Location Loc) {
  return Builder.create<arith::ConstantOp>(Loc, Type,
                                            cast<TypedAttr>(Value));
}

LogicalResult BinaryOp::verify() {
  StringRef Operator = getOperatorName();
  if (Operator.size() != 1)
    return emitOpError("requires a one-character operator");
  if (Operator == "=")
    return emitOpError("does not represent the assignment operator");
  return success();
}

OpFoldResult BinaryOp::fold(FoldAdaptor Adaptor) {
  auto LHS = dyn_cast_or_null<FloatAttr>(Adaptor.getLhs());
  auto RHS = dyn_cast_or_null<FloatAttr>(Adaptor.getRhs());
  if (!LHS || !RHS)
    return {};

  APFloat Result = LHS.getValue();
  StringRef Operator = getOperatorName();
  if (Operator == "+")
    Result.add(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else if (Operator == "-")
    Result.subtract(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else if (Operator == "*")
    Result.multiply(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else
    return {};

  return FloatAttr::get(getResult().getType(), Result);
}
