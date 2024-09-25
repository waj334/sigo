#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"

namespace mlir::go
{
struct GlobalConstantsPass
  : public mlir::PassWrapper<GlobalConstantsPass, mlir::OperationPass<mlir::ModuleOp>>
{
  void runOnOperation() final
  {
    auto module = getOperation();
    mlir::DataLayout dataLayout(module);

    mlir::LowerToLLVMOptions options(&getContext(), dataLayout);
    if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout"));
        dataLayoutStr)
    {
      llvm::DataLayout llvmDataLayout(dataLayoutStr);
      options.dataLayout = llvmDataLayout;
    }

    mlir::go::LLVMTypeConverter converter(module, options);

    auto _stringType = converter.convertType(converter.lookupRuntimeType("string"));

    // Collect the values that globals will be created from
    SmallVector<std::pair<std::string, mlir::Location>> globalStrings;
    module.walk(
      [&](ConstantOp constOp)
      {
        if (go::isa<StringType>(constOp.getType()))
        {
          const auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constOp.getValue());

          // Has this string already been encountered?
          auto it = std::find_if(
            globalStrings.begin(),
            globalStrings.end(),
            [&](const std::pair<std::string, mlir::Location>& value)
            { return value.first == strAttr.getValue().str(); });
          if (it == globalStrings.end())
          {
            // Not found. Add it
            globalStrings.push_back(std::make_pair(strAttr.getValue().str(), constOp.getLoc()));
          }
        }
      });

    // Create the builder
    OpBuilder builder(module.getBodyRegion());

    // Create the global constant strings
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      for (const auto& [str, loc] : globalStrings)
      {
        const auto strHash = hash_value(llvm::StringRef(str));

        // Create the global Go-string
        std::string name = "gostr_" + std::to_string(strHash);
        mlir::LLVM::GlobalOp globalGoStrOp;
        globalGoStrOp = builder.create<mlir::LLVM::GlobalOp>(
          loc, _stringType, true, mlir::LLVM::Linkage::External, name, Attribute(), 0);

        // Init block
        {
          mlir::OpBuilder::InsertionGuard guard(builder);

          // Create the initializer block
          auto initBlock = builder.createBlock(&globalGoStrOp.getInitializerRegion());

          // Position the build inside the init block
          builder.setInsertionPointToStart(initBlock);

          // Create the global C-string
          name = "cstr_" + std::to_string(strHash);
          auto globalCStr =
            mlir::LLVM::createGlobalString(loc, builder, name, str, mlir::LLVM::Linkage::External);

          // Position the global after the C-string
          builder.setInsertionPointAfterValue(globalCStr);

          // Create the string length constant value
          mlir::Value lenVal = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(str.length()));

          // Create the struct
          mlir::Value structValue = builder.create<mlir::LLVM::UndefOp>(loc, _stringType);
          structValue = builder.create<mlir::LLVM::InsertValueOp>(loc, structValue, globalCStr, 0);
          structValue = builder.create<mlir::LLVM::InsertValueOp>(loc, structValue, lenVal, 1);
          builder.create<mlir::LLVM::ReturnOp>(loc, structValue);
        }
      }
    }

    // Declare all other globals
    createGlobals(module, converter, builder);
  }

  void
  createGlobals(mlir::ModuleOp module, const LLVMTypeConverter& typeConverter, OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    DenseMap<StringRef, GlobalOp> globalOps;

    // Find all globals. The resulting set is de-duplicated by symbol name.
    module.walk(
      [&](GlobalOp op)
      {
        const auto symbol = op.getSymName();

        // Avoid redefinitions.
        if (module.lookupSymbol(symbol))
          return;

        // Has this global already been encountered?
        if (globalOps.contains(op.getSymName()))
          return;

        // Not found. Add it.
        globalOps[op.getSymName()] = op;
      });

    // Declare each of the globals.
    builder.setInsertionPointToStart(module.getBody());
    for (auto [symbol, op] : globalOps)
    {
      auto type = go::cast<PointerType>(op.getType());
      auto resultType = typeConverter.convertType(*type.getElementType());

      // TODO: Alignment needs to be specified to avoid issues on platforms that do not allow
      // unaligned memory access.
      auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
        op.getLoc(),
        resultType,
        false,
        mlir::LLVM::Linkage::External,
        op.getSymName(),
        Attribute());

      if (op->hasAttr("llvm.debug.global_expr"))
      {
        auto globalExpr =
          mlir::cast<LLVM::DIGlobalVariableExpressionAttr>(op->getAttr("llvm.debug.global_expr"));
        globalOp.setDbgExprAttr(globalExpr);
      }

      bool shouldInitialize = true;

      if (op->hasAttr("llvm.linkage"))
      {
        if (auto linkage = mlir::cast<LLVM::LinkageAttr>(op->getAttr("llvm.linkage"));
            linkage.getLinkage() == LLVM::Linkage::External)
        {
          // Explicit external global should not be considered for initialization as externally
          // linked objects are expected to satisfy this dependency.
          shouldInitialize = false;
        }
      }

      // Create the init block only if the global should be initialized
      if (shouldInitialize)
      {
        mlir::OpBuilder::InsertionGuard guard(builder);

        // Create the initializer block
        auto initBlock = builder.createBlock(&globalOp.getInitializerRegion());

        // Position the build inside the init block
        builder.setInsertionPointToStart(initBlock);

        // Zero initialize the global for now
        // TODO: The usages of the global should be analyzed. The package init methods contain
        // initializers
        //       for these globals and the foldable and materializable operations could be moved
        //       here in theory.

        mlir::Value zeroValue = builder.create<mlir::LLVM::ZeroOp>(op.getLoc(), resultType);
        builder.create<mlir::LLVM::ReturnOp>(op.getLoc(), zeroValue);
      }
    }
  }

  StringRef getArgument() const final { return "go-global-constants-pass"; }

  StringRef getDescription() const final { return "Create global constants"; }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
};

std::unique_ptr<mlir::Pass> createGlobalConstantsPass()
{
  return std::make_unique<GlobalConstantsPass>();
}
} // namespace mlir::go
