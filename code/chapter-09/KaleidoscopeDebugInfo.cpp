#include "KaleidoscopeDebugInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace {

using FunctionParameterMap = std::map<std::string, std::vector<std::string>>;

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

    StringRef inputPath = inputFilename;
    auto file = inputPath.empty()
                    ? LLVM::DIFileAttr::get(context, "<stdin>", "")
                    : LLVM::DIFileAttr::get(
                          context, llvm::sys::path::filename(inputPath),
                          llvm::sys::path::parent_path(inputPath));
    auto compileUnit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C,
        file, StringAttr::get(context, "Kaleidoscope"),
        /*isOptimized=*/optLevel != '0', LLVM::DIEmissionKind::Full);
    module->setLoc(FusedLoc::get(context, {module.getLoc()}, compileUnit));

    auto doubleType =
        LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                   "double", 64, llvm::dwarf::DW_ATE_float);
    auto emptyExpression = LLVM::DIExpressionAttr::get(context);

    for (LLVM::LLVMFuncOp function : module.getOps<LLVM::LLVMFuncOp>()) {
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

      auto functionType = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {});
      auto name = function.getNameAttr();
      auto scope = LLVM::DISubprogramAttr::get(
          context, id, functionCompileUnit, functionFile, name, name,
          functionFile, line, line, flags, functionType,
          /*retainedNodes=*/{}, /*annotations=*/{});
      function->setLoc(FusedLoc::get(context, {originalLoc}, scope));

      auto names = functionParameters.find(function.getName().str());
      if (function.isExternal() || names == functionParameters.end())
        continue;

      Block &entryBlock = function.getBody().front();
      for (auto [argumentNumber, parameterName] :
           llvm::enumerate(names->second)) {
        if (argumentNumber >= entryBlock.getNumArguments())
          break;

        BlockArgument argument = entryBlock.getArgument(argumentNumber);
        LLVM::StoreOp argumentStore;
        for (LLVM::StoreOp store : entryBlock.getOps<LLVM::StoreOp>()) {
          if (store.getValue() == argument) {
            argumentStore = store;
            break;
          }
        }
        if (!argumentStore)
          continue;

        auto variable = LLVM::DILocalVariableAttr::get(
            scope, parameterName, scope.getFile(), scope.getLine(),
            argumentNumber + 1, /*alignInBits=*/0, doubleType,
            LLVM::DIFlags::Zero);
        Value variableAddress = findUnderlyingAlloca(argumentStore.getAddr());
        if (!variableAddress)
          variableAddress = argumentStore.getAddr();

        OpBuilder builder(argumentStore);
        builder.setInsertionPointAfter(argumentStore);
        builder.create<LLVM::DbgDeclareOp>(
            argumentStore.getLoc(), variableAddress, variable, emptyExpression);
      }
    }
  }

private:
  std::string inputFilename;
  char optLevel;
  FunctionParameterMap functionParameters;
};

} // namespace

std::unique_ptr<Pass> createKaleidoscopeDebugInfoPass(
    StringRef inputFilename, char optLevel,
    const std::map<std::string, std::vector<std::string>> &functionParameters) {
  return std::make_unique<KaleidoscopeDebugInfoPass>(inputFilename, optLevel,
                                                     functionParameters);
}
