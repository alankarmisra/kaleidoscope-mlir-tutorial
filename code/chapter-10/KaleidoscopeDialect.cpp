// KaleidoscopeDialect.cpp

#include "KaleidoscopeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kaleidoscope;

#include "KaleidoscopeDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "KaleidoscopeTypes.cpp.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.cpp.inc"

void KaleidoscopeDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KaleidoscopeTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "KaleidoscopeOps.cpp.inc"
      >();
}
