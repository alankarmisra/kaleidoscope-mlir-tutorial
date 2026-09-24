// KaleidoscopeDialect.cpp

#include "KaleidoscopeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kaleidoscope;

// Generated definitions for the dialect class declared by the matching header
// fragment.
#include "KaleidoscopeDialect.cpp.inc"

// Select the generated C++ definitions for our TypeDef and Op records.
#define GET_TYPEDEF_CLASSES
#include "KaleidoscopeTypes.cpp.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.cpp.inc"

void KaleidoscopeDialect::initialize() {
  // The same generated .inc files also contain lists of every type and
  // operation in the dialect. These macros select those lists so the dialect
  // can register all generated classes with MLIR.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KaleidoscopeTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "KaleidoscopeOps.cpp.inc"
      >();
}
