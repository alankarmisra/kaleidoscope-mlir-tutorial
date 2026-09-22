#ifndef KALEIDOSCOPE_DIALECT_H
#define KALEIDOSCOPE_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "KaleidoscopeDialect.h.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.h.inc"

#endif
