#ifndef KALEIDOSCOPE_DEBUG_INFO_H
#define KALEIDOSCOPE_DEBUG_INFO_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

std::unique_ptr<mlir::Pass> createKaleidoscopeDebugInfoPass(
    llvm::StringRef inputFilename, char optLevel,
    const std::map<std::string, std::vector<std::string>> &functionParameters);

#endif
