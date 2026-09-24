#include "KaleidoscopeDebugInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace {

using FunctionParameterMap = std::map<std::string, std::vector<std::string>>;

/// Walk backwards from a value produced during lowering until we reach the
/// llvm.alloca that owns its stack storage.
///
/// A debug declaration describes the storage for a variable, not a temporary
/// value derived from it. Lowering may insert address calculations, bitcasts,
/// or other operations between the source variable and its alloca, so we
/// follow the operand chain until we find the alloca itself. Returns a null
/// Value if no alloca is found.
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

    // A DWARF file entry is a pair of (directory, filename), not a single
    // path. Input read interactively from the REPL has no real path at all,
    // so we invent a synthetic <stdin> file to describe it.
    StringRef inputPath = inputFilename;
    auto file = inputPath.empty()
                    ? LLVM::DIFileAttr::get(context, "<stdin>", "")
                    : LLVM::DIFileAttr::get(
                          context, llvm::sys::path::filename(inputPath),
                          llvm::sys::path::parent_path(inputPath));

    // The compile unit is the top-level debug record for this translation
    // unit. Wrapping it in a DistinctAttr guarantees it will not be merged
    // with another compile unit that happens to have identical fields, which
    // would be a DWARF-validity bug.
    auto compileUnit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C,
        file, StringAttr::get(context, "Kaleidoscope"),
        /*isOptimized=*/optLevel != '0', LLVM::DIEmissionKind::Full);

    // MLIR carries debug metadata on locations rather than on a separate
    // side table. Fusing the compile unit into the module's location is how
    // we attach it so that LLVM IR translation can recover it later.
    module->setLoc(FusedLoc::get(context, {module.getLoc()}, compileUnit));

    // Kaleidoscope has exactly one source-level type today. Every parameter
    // we describe below uses this same DWARF base type.
    auto doubleType =
        LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                   "double", 64, llvm::dwarf::DW_ATE_float);
    auto emptyExpression = LLVM::DIExpressionAttr::get(context);

    for (LLVM::LLVMFuncOp function : module.getOps<LLVM::LLVMFuncOp>()) {
      // Choose the DIFile and source line for this function, preferring the
      // information recorded on the function's source location. When the
      // parser supplied no concrete location, fall back to the module's input
      // file and line 1.
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
      // distinct identity so it will not be merged with an identical-looking
      // subprogram, and is marked as a Definition. A declaration has no body
      // emitted by this compile unit, so it gets neither attachment.
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

      // Kaleidoscope has no source-level function signatures yet, so we
      // cannot describe parameter or return types accurately here. An empty
      // DISubroutineType is the minimum required to attach a DISubprogram;
      // the parameters themselves are described later by DILocalVariable
      // entries on their stack slots.
      auto functionType = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {});
      auto name = function.getNameAttr();
      auto scope = LLVM::DISubprogramAttr::get(
          context, id, functionCompileUnit, functionFile, name, name,
          functionFile, line, line, flags, functionType,
          /*retainedNodes=*/{}, /*annotations=*/{});

      // Fuse the subprogram scope into the function's existing location.
      // Later lowering and LLVM IR translation will use this scope for
      // instructions emitted inside the function body.
      function->setLoc(FusedLoc::get(context, {originalLoc}, scope));

      // Declarations have no body, and a definition may be absent from the
      // parser-side parameter table when it was generated by the compiler
      // rather than written by the user. In the latter case, the function may
      // still have parameters, but this pass does not know their source names.
      auto names = functionParameters.find(function.getName().str());
      if (function.isExternal() || names == functionParameters.end())
        continue;

      Block &entryBlock = function.getBody().front();
      for (auto [argumentNumber, parameterName] :
           llvm::enumerate(names->second)) {
        if (argumentNumber >= entryBlock.getNumArguments())
          break;

        BlockArgument argument = entryBlock.getArgument(argumentNumber);

        // A mutable Kaleidoscope parameter is copied into a stack slot at
        // the top of the entry block. Locate the store of the incoming
        // argument so we can point the debug declaration at the stack slot
        // rather than at the argument itself.
        LLVM::StoreOp argumentStore;
        for (LLVM::StoreOp store : entryBlock.getOps<LLVM::StoreOp>()) {
          if (store.getValue() == argument) {
            argumentStore = store;
            break;
          }
        }
        if (!argumentStore)
          continue;

        // DWARF argument numbers are one-based. The source name comes from
        // the parser-side table because LLVM block arguments do not retain
        // it. The scope, file, and line are inherited from the enclosing
        // function.
        auto variable = LLVM::DILocalVariableAttr::get(
            scope, parameterName, scope.getFile(), scope.getLine(),
            argumentNumber + 1, /*alignInBits=*/0, doubleType,
            LLVM::DIFlags::Zero);

        // The store's address operand may be an intermediate value rather
        // than the alloca itself. Walk back to the alloca if one exists;
        // otherwise, fall back to the store's own address.
        Value variableAddress = findUnderlyingAlloca(argumentStore.getAddr());
        if (!variableAddress)
          variableAddress = argumentStore.getAddr();

        // llvm.dbg.declare attaches the debug variable to the storage
        // address. It emits no executable code — it is a hint for the
        // debugger, translated to a #dbg_declare record in LLVM IR.
        OpBuilder builder(argumentStore);
        builder.setInsertionPointAfter(argumentStore);
        builder.create<LLVM::DbgDeclareOp>(
            argumentStore.getLoc(), variableAddress, variable, emptyExpression);
      }
    }
  }

private:
  // The pass owns copies of these because the caller's strings and maps are
  // not guaranteed to outlive execution of the pass manager.
  std::string inputFilename;
  char optLevel;
  FunctionParameterMap functionParameters;
};

} // namespace

// Construction is hidden behind a factory so toy.cpp does not need to know
// the concrete pass implementation type.
std::unique_ptr<Pass> createKaleidoscopeDebugInfoPass(
    StringRef inputFilename, char optLevel,
    const std::map<std::string, std::vector<std::string>> &functionParameters) {
  return std::make_unique<KaleidoscopeDebugInfoPass>(inputFilename, optLevel,
                                                     functionParameters);
}
