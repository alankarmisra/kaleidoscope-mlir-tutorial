// KaleidoscopeDebugInfo.h

#ifndef KALEIDOSCOPE_DEBUG_INFO_H
#define KALEIDOSCOPE_DEBUG_INFO_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

std::unique_ptr<mlir::Pass>
createKaleidoscopeDebugInfoPass(llvm::StringRef inputFilename, char optLevel);

#endif
