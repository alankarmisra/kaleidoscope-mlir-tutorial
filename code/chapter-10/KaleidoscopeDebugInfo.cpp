// KaleidoscopeDebugInfo.cpp
//
// Constructs the compile-unit and function-scope debug metadata that later
// lowering and LLVM IR translation will attach to instructions. Variable-
// level debug info is produced separately by the variable-lowering pass.

#include "KaleidoscopeDebugInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace {

class KaleidoscopeDebugInfoPass
    : public PassWrapper<KaleidoscopeDebugInfoPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KaleidoscopeDebugInfoPass)

  KaleidoscopeDebugInfoPass(StringRef inputFilename, char optLevel)
      : inputFilename(inputFilename.str()), optLevel(optLevel) {}

  StringRef getArgument() const final { return "kaleidoscope-debug-info"; }
  StringRef getDescription() const final {
    return "Add Kaleidoscope compile-unit and function debug info";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    // A DWARF file entry is a (directory, filename) pair, not a single path.
    // Input read interactively from the REPL has no real path, so we invent
    // a synthetic <stdin> entry to describe it.
    StringRef inputPath = inputFilename;
    auto file = inputPath.empty()
                    ? LLVM::DIFileAttr::get(context, "<stdin>", "")
                    : LLVM::DIFileAttr::get(
                          context, llvm::sys::path::filename(inputPath),
                          llvm::sys::path::parent_path(inputPath));

    // Create the top-level debug record for this translation unit. We wrap
    // it in a DistinctAttr so it has identity: two compile units with
    // otherwise identical fields must not be uniqued into one, which would
    // be invalid DWARF.
    auto compileUnit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C,
        file, StringAttr::get(context, "Kaleidoscope"),
        /*isOptimized=*/optLevel != '0', LLVM::DIEmissionKind::Full);

    // MLIR carries debug metadata on locations rather than in a separate
    // side table. Fusing the compile unit into the module's existing
    // location is how we attach it so that LLVM IR translation can recover
    // both the original source location and the new metadata.
    module->setLoc(FusedLoc::get(context, {module.getLoc()}, compileUnit));

    for (LLVM::LLVMFuncOp function : module.getOps<LLVM::LLVMFuncOp>()) {
      // Prefer the function's own file and line. Fall back to the module's
      // input file and line 1 for compiler-generated functions that carry
      // no concrete source location.
      Location originalLoc = function.getLoc();
      LLVM::DIFileAttr functionFile = file;
      int64_t line = 1;
      if (auto fileLoc = originalLoc->findInstanceOf<FileLineColLoc>()) {
        StringRef functionPath = fileLoc.getFilename().getValue();
        functionFile = LLVM::DIFileAttr::get(
            context, llvm::sys::path::filename(functionPath),
            llvm::sys::path::parent_path(functionPath));
        line = fileLoc.getLine();
      }

      // A definition and an external declaration need different DISubprogram
      // configurations. A definition belongs to this compile unit, carries a
      // distinct identity so it won't be merged with an identical-looking
      // subprogram, and is flagged as a Definition. A declaration has no
      // body emitted by this compile unit, so it gets neither the attachment
      // nor the flag.
      DistinctAttr id;
      LLVM::DICompileUnitAttr functionCompileUnit = compileUnit;
      auto flags = static_cast<LLVM::DISubprogramFlags>(0);
      if (optLevel != '0')
        flags = flags | LLVM::DISubprogramFlags::Optimized;
      if (function.isExternal()) {
        functionCompileUnit = {};
      } else {
        id = DistinctAttr::create(UnitAttr::get(context));
        flags = flags | LLVM::DISubprogramFlags::Definition;
      }

      // Create the function's lexical debug scope. A complete source-level
      // function signature is outside this tutorial's scope, but a
      // DISubprogram still requires a subroutine-type metadata node, so we
      // supply an empty one. Individual parameter variables receive their
      // types later through their debug declarations; the function signature
      // itself remains unspecified.
      auto functionType = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {});
      auto name = function.getNameAttr();
      auto scope = LLVM::DISubprogramAttr::get(
          context, id, functionCompileUnit, functionFile, name, name,
          functionFile, line, line, flags, functionType,
          /*retainedNodes=*/{}, /*annotations=*/{});

      // Preserve the original source location while attaching the subprogram
      // scope. Subsequent debug-info lowering and LLVM IR translation use
      // this scope when they need to describe instructions in the function
      // body, so it must be reachable from the function's location.
      function->setLoc(FusedLoc::get(context, {originalLoc}, scope));
    }
  }

private:
  // The pass owns copies of these because execution of the pass manager is
  // not guaranteed to finish within the lifetime of the caller's arguments.
  std::string inputFilename;
  char optLevel;
};

} // namespace

// Hide the concrete pass class behind a factory so the driver only needs the
// standard mlir::Pass interface.
std::unique_ptr<Pass> createKaleidoscopeDebugInfoPass(StringRef inputFilename,
                                                      char optLevel) {
  return std::make_unique<KaleidoscopeDebugInfoPass>(inputFilename, optLevel);
}
