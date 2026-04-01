#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"
#include <Go/IR/GoTypes.h>

namespace mlir::go
{
#define GEN_PASS_DEF_GLOBALCONSTANTSPASS
#include "Go/Transforms/Passes.h.inc"

struct GlobalConstantsPass : public impl::GlobalConstantsPassBase<GlobalConstantsPass>
{
  using GlobalConstantsPassBase<GlobalConstantsPass>::GlobalConstantsPassBase;

  void runOnOperation() final
  {
    auto module = getOperation();
    mlir::DataLayout dataLayout(module);

    // Create the builder
    OpBuilder builder(module.getBodyRegion());

    mlir::LowerToLLVMOptions options(&getContext(), dataLayout);
    if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout"));
        dataLayoutStr)
    {
      llvm::DataLayout llvmDataLayout(dataLayoutStr);
      options.dataLayout = llvmDataLayout;
    }

    mlir::go::LLVMTypeConverter converter(module, options);

    auto _stringType = converter.convertType(converter.lookupRuntimeType("string"));

    // Fold all constants with a reference or a body and then replace the operation.
    module.walk(
      [&](ConstantOp op)
      {
        if ((!op.getSymRef() && op.getBody().empty()) || op.getValue())
        {
          return;
        }

        SmallVector<OpFoldResult, 4> foldResults;
        if (failed(op->fold(foldResults)) || foldResults.empty())
        {
          op->emitOpError("failed to fold constant");
          signalPassFailure();
          return;
        }

        const auto foldedValue = mlir::dyn_cast<mlir::Attribute>(foldResults[0]);
        if (!foldedValue)
        {
          op->emitOpError("fold did not produce an attribute");
          signalPassFailure();
          return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);

        // Create a new constant operation returning the folded value.
        auto newConstOp =
          ConstantOp::create(builder, op.getLoc(), op.getType(), foldedValue, StringAttr());

        // Replace the old operation.
        op->replaceAllUsesWith(newConstOp);

        // Remove the old operation.
        op.erase();
      });

    // Now all global constants can be removed.
    module.walk(
      [&](GlobalConstantOp op)
      {
        if (const auto fusedLoc = op.getLoc()->findInstanceOf<mlir::FusedLoc>())
        {
          if (
            const auto diGlobalExprAttr =
              mlir::dyn_cast_or_null<mlir::LLVM::DIGlobalVariableExpressionAttr>(
                fusedLoc.getMetadata()))
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(module.getBody());

            mlir::Type elementT;

            if (op.getValue())
            {
              elementT = mlir::TypeSwitch<mlir::Attribute, mlir::Type>(*op.getValue())
                           .Case([&](const mlir::go::ComplexNumberAttr attr)
                                 { return mlir::ComplexType::get(attr.getImag().getType()); })
                           .Case([&](const mlir::FloatAttr attr) { return attr.getType(); })
                           .Case(
                             [&](const mlir::IntegerAttr attr) -> mlir::Type
                             {
                               if (mlir::cast<mlir::IntegerType>(attr.getType()).getWidth() == 1)
                               {
                                 return mlir::go::BooleanType::get(&getContext());
                               }
                               return mlir::go::IntegerType::get(
                                 &getContext(), mlir::go::IntegerType::Signed);
                             })
                           .Case([&](const mlir::StringAttr)
                                 { return mlir::go::StringType::get(&getContext()); });
            }

            if (elementT)
            {
              auto resultType = converter.convertType(elementT);

              auto globalOp = mlir::LLVM::GlobalOp::create(builder, 
                op.getLoc(),
                resultType,
                true,
                mlir::LLVM::Linkage::Internal,
                op.getSymName(),
                Attribute());

              globalOp.setDbgExprsAttr(
                mlir::ArrayAttr::get(
                  module->getContext(), SmallVector<mlir::Attribute>{ diGlobalExprAttr }));

              {
                mlir::OpBuilder::InsertionGuard initGuard(builder);
                auto initBlock = builder.createBlock(&globalOp.getInitializerRegion());
                builder.setInsertionPointToStart(initBlock);
                mlir::Value undefValue =
                  mlir::LLVM::UndefOp::create(builder, op.getLoc(), resultType);
                mlir::LLVM::ReturnOp::create(builder, op.getLoc(), undefValue);
              }
            }
          }
        }

        op.erase();
      });

    // Collect the values that globals will be created from.
    SmallVector<std::pair<std::string, mlir::Location>> globalStrings;
    module.walk(
      [&](ConstantOp constOp)
      {
        if (go::isa<go::StringType>(constOp.getType()))
        {
          const auto strAttr = mlir::dyn_cast<mlir::StringAttr>(*constOp.getValue());

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

    // Create the global constant strings
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&module.getBodyRegion().front());
      for (const auto& [str, loc] : globalStrings)
      {
        const auto strHash = hash_value(llvm::StringRef(str));

        // Create the global Go-string
        std::string name = "gostr_" + std::to_string(strHash);
        mlir::LLVM::GlobalOp globalGoStrOp;
        globalGoStrOp = mlir::LLVM::GlobalOp::create(builder, 
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
          mlir::Value lenVal = mlir::LLVM::ConstantOp::create(builder, 
            loc, builder.getI32Type(), builder.getI32IntegerAttr(str.length()));

          // Create the struct
          mlir::Value structValue = mlir::LLVM::UndefOp::create(builder, loc, _stringType);
          structValue = mlir::LLVM::InsertValueOp::create(builder, loc, structValue, globalCStr, ArrayRef<int64_t>{0});
          structValue = mlir::LLVM::InsertValueOp::create(builder, loc, structValue, lenVal, ArrayRef<int64_t>{1});
          mlir::LLVM::ReturnOp::create(builder, loc, structValue);
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
      auto globalOp = mlir::LLVM::GlobalOp::create(builder, 
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
        globalOp.setDbgExprsAttr(
          mlir::ArrayAttr::get(module->getContext(), SmallVector<mlir::Attribute>{ globalExpr }));
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

        mlir::Value zeroValue = mlir::LLVM::ZeroOp::create(builder, op.getLoc(), resultType);
        mlir::LLVM::ReturnOp::create(builder, op.getLoc(), zeroValue);
      }
    }
  }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
};

} // namespace mlir::go
