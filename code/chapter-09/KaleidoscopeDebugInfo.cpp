#include "KaleidoscopeDebugInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace {

using FunctionParameterMap = std::map<std::string, std::vector<std::string>>;

/// Follow the address-producing operations inserted during lowering until we
/// reach the llvm.alloca that owns the variable's stack storage. A debug
/// declaration describes the variable's storage, not an intermediate cast or
/// address calculation derived from it.
static Value findUnderlyingAlloca(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return {};
  if (isa<LLVM::AllocaOp>(definingOp))
    return value;
  for (Value operand : definingOp->getOperands())
    if (Value alloca = findUnderlyingAlloca(operand))
      return alloca;
  return {};
}

class KaleidoscopeDebugInfoPass
    : public PassWrapper<KaleidoscopeDebugInfoPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KaleidoscopeDebugInfoPass)

  KaleidoscopeDebugInfoPass(StringRef inputFilename, char optLevel,
                            const FunctionParameterMap &functionParameters)
      : inputFilename(inputFilename.str()), optLevel(optLevel),
        functionParameters(functionParameters) {}

  StringRef getArgument() const final { return "kaleidoscope-debug-info"; }
  StringRef getDescription() const final {
    return "Add Kaleidoscope compile-unit, function, and parameter debug info";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    // DWARF describes source files using a filename and directory separately.
    // Input read from the REPL has no real path, so use a synthetic <stdin>
    // file instead.
    StringRef inputPath = inputFilename;
    auto file = inputPath.empty()
                    ? LLVM::DIFileAttr::get(context, "<stdin>", "")
                    : LLVM::DIFileAttr::get(
                          context, llvm::sys::path::filename(inputPath),
                          llvm::sys::path::parent_path(inputPath));

    // The compile unit is the top-level debug record for this translation
    // unit. DistinctAttr gives it identity: two otherwise identical compile
    // units must not be merged merely because their fields compare equal.
    auto compileUnit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C,
        file, StringAttr::get(context, "Kaleidoscope"),
        /*isOptimized=*/optLevel != '0', LLVM::DIEmissionKind::Full);

    // MLIR carries debug metadata in locations. Preserve the module's original
    // location and fuse the compile-unit metadata into it so translation to
    // LLVM IR can recover both pieces of information.
    module->setLoc(FusedLoc::get(context, {module.getLoc()}, compileUnit));

    // All Kaleidoscope values currently have the same source-level type.
    // Reuse this metadata when describing each function parameter.
    auto doubleType =
        LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                   "double", 64, llvm::dwarf::DW_ATE_float);
    auto emptyExpression = LLVM::DIExpressionAttr::get(context);

    for (LLVM::LLVMFuncOp function : module.getOps<LLVM::LLVMFuncOp>()) {
      // Prefer the function's own source file and line. Fall back to the
      // module input and line 1 when the parser supplied no concrete location.
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

      // A definition owns a distinct DISubprogram and belongs to this compile
      // unit. An external declaration has no body emitted by this compile unit,
      // so it is described without a compile-unit attachment or Definition
      // flag.
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

      // Kaleidoscope has only double-valued functions, but this tutorial does
      // not yet encode a complete DWARF function signature. The subroutine type
      // still establishes the metadata node required by DISubprogram.
      auto functionType = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {});
      auto name = function.getNameAttr();
      auto scope = LLVM::DISubprogramAttr::get(
          context, id, functionCompileUnit, functionFile, name, name,
          functionFile, line, line, flags, functionType,
          /*retainedNodes=*/{}, /*annotations=*/{});

      // Fuse the subprogram scope into the existing source location. Later
      // lowering and LLVM IR translation use this scope for instructions in
      // the function body.
      function->setLoc(FusedLoc::get(context, {originalLoc}, scope));

      // Declarations have no entry block or local variables. A definition may
      // also be absent from the side table when it was compiler-generated.
      auto names = functionParameters.find(function.getName().str());
      if (function.isExternal() || names == functionParameters.end())
        continue;

      Block &entryBlock = function.getBody().front();
      for (auto [argumentNumber, parameterName] :
           llvm::enumerate(names->second)) {
        if (argumentNumber >= entryBlock.getNumArguments())
          break;

        BlockArgument argument = entryBlock.getArgument(argumentNumber);

        // Mutable Kaleidoscope parameters are copied into stack slots during
        // lowering. Find the store of this incoming argument so we can connect
        // the source variable to the address that subsequently holds it.
        LLVM::StoreOp argumentStore;
        for (LLVM::StoreOp store : entryBlock.getOps<LLVM::StoreOp>()) {
          if (store.getValue() == argument) {
            argumentStore = store;
            break;
          }
        }
        if (!argumentStore)
          continue;

        // DWARF argument numbers are one-based. The source name comes from the
        // parser-side table because LLVM block arguments do not retain it.
        auto variable = LLVM::DILocalVariableAttr::get(
            scope, parameterName, scope.getFile(), scope.getLine(),
            argumentNumber + 1, /*alignInBits=*/0, doubleType,
            LLVM::DIFlags::Zero);
        Value variableAddress = findUnderlyingAlloca(argumentStore.getAddr());
        if (!variableAddress)
          variableAddress = argumentStore.getAddr();

        // llvm.dbg.declare tells the debugger that this address is the storage
        // for the named source parameter. It does not generate executable code.
        OpBuilder builder(argumentStore);
        builder.setInsertionPointAfter(argumentStore);
        builder.create<LLVM::DbgDeclareOp>(
            argumentStore.getLoc(), variableAddress, variable, emptyExpression);
      }
    }
  }

private:
  // Pass instances own their options because the caller's strings and maps may
  // not outlive execution of the pass manager.
  std::string inputFilename;
  char optLevel;
  FunctionParameterMap functionParameters;
};

} // namespace

// Keep construction behind a small factory so toy.cpp does not need to know
// the concrete pass implementation type.
std::unique_ptr<Pass> createKaleidoscopeDebugInfoPass(
    StringRef inputFilename, char optLevel,
    const std::map<std::string, std::vector<std::string>> &functionParameters) {
  return std::make_unique<KaleidoscopeDebugInfoPass>(inputFilename, optLevel,
                                                     functionParameters);
}
