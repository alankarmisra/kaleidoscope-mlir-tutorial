// KaleidoscopeDebugInfo.h

#ifndef KALEIDOSCOPE_DEBUG_INFO_H
#define KALEIDOSCOPE_DEBUG_INFO_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

/// Create the module pass that attaches compile-unit and function debug scopes
/// before the LLVM dialect is translated to LLVM IR.
///
/// Chapter 10 no longer needs a separate parameter-name map here: source
/// variable names are preserved by the Kaleidoscope dialect and consumed by
/// its lowering pass.
std::unique_ptr<mlir::Pass>
createKaleidoscopeDebugInfoPass(llvm::StringRef inputFilename, char optLevel);

#endif
