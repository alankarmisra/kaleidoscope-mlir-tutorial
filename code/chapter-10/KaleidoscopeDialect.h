// KaleidoscopeDialect.h

#ifndef KALEIDOSCOPE_DIALECT_H
#define KALEIDOSCOPE_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated declaration of mlir::kaleidoscope::KaleidoscopeDialect.
#include "KaleidoscopeDialect.h.inc"

// KaleidoscopeTypes.h.inc contains several selectable sections. Defining this
// macro asks it to emit the generated type class declarations at this include.
#define GET_TYPEDEF_CLASSES
#include "KaleidoscopeTypes.h.inc"

// Likewise, select the generated operation class declarations from the
// operation header fragment.
#define GET_OP_CLASSES
#include "KaleidoscopeOps.h.inc"

#endif
