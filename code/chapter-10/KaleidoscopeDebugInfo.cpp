// KaleidoscopeDebugInfo.cpp

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

    // DWARF stores the source filename and directory separately. Use a
    // synthetic file when the compiler is reading interactively from stdin.
    StringRef inputPath = inputFilename;
    auto file = inputPath.empty()
                    ? LLVM::DIFileAttr::get(context, "<stdin>", "")
                    : LLVM::DIFileAttr::get(
                          context, llvm::sys::path::filename(inputPath),
                          llvm::sys::path::parent_path(inputPath));

    // Create the top-level debug record for this translation unit. It is
    // distinct because compile units have identity and must not be uniqued with
    // another compile unit that happens to contain the same fields.
    auto compileUnit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C,
        file, StringAttr::get(context, "Kaleidoscope"),
        /*isOptimized=*/optLevel != '0', LLVM::DIEmissionKind::Full);

    // Debug metadata travels through MLIR as location metadata. Fusing the
    // compile unit with the existing module location preserves both.
    module->setLoc(FusedLoc::get(context, {module.getLoc()}, compileUnit));

    for (LLVM::LLVMFuncOp function : module.getOps<LLVM::LLVMFuncOp>()) {
      // Use the function's source file and line when available, with the module
      // file and line 1 as the fallback for unknown locations.
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

      // Definitions belong to this compile unit and require a distinct
      // DISubprogram identity. External declarations have neither a body nor a
      // definition emitted by this compile unit.
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
      // function type is outside this tutorial's scope, but DISubprogram still
      // requires a subroutine-type metadata node.
      auto functionType = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {});
      auto name = function.getNameAttr();
      auto scope = LLVM::DISubprogramAttr::get(
          context, id, functionCompileUnit, functionFile, name, name,
          functionFile, line, line, flags, functionType,
          /*retainedNodes=*/{}, /*annotations=*/{});

      // Preserve the original source location while attaching the scope used
      // by subsequent debug-info lowering and LLVM IR translation.
      function->setLoc(FusedLoc::get(context, {originalLoc}, scope));
    }
  }

private:
  // Store owned copies because pass execution may outlive the caller's values.
  std::string inputFilename;
  char optLevel;
};

} // namespace

// Hide the concrete pass class from the driver and expose only the standard
// mlir::Pass interface.
std::unique_ptr<Pass> createKaleidoscopeDebugInfoPass(StringRef inputFilename,
                                                      char optLevel) {
  return std::make_unique<KaleidoscopeDebugInfoPass>(inputFilename, optLevel);
}
