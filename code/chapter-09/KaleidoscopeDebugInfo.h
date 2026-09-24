#ifndef KALEIDOSCOPE_DEBUG_INFO_H
#define KALEIDOSCOPE_DEBUG_INFO_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

/// Create the module pass that adds the debug metadata needed before the LLVM
/// dialect is translated to LLVM IR.
///
/// `inputFilename` identifies the source file (or is empty for stdin), and
/// `optLevel` determines whether the compile unit is marked as optimized.
/// `functionParameters` preserves source parameter names, which are no longer
/// present in the lowered LLVM function arguments themselves.
std::unique_ptr<mlir::Pass> createKaleidoscopeDebugInfoPass(
    llvm::StringRef inputFilename, char optLevel,
    const std::map<std::string, std::vector<std::string>> &functionParameters);

#endif
