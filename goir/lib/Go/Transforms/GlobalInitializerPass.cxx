#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Target/LLVMIR/TypeToLLVM.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

constexpr int32_t packageInitBasePriority = 1000000000;

namespace mlir::go
{
struct GlobalInitializerPass : PassWrapper<GlobalInitializerPass, OperationPass<ModuleOp>>
{
  SmallVector<Operation*> opsToRemove;

  StringRef getArgument() const final { return "global-initializer-pass"; }

  StringRef getDescription() const final
  {
    return "Global Initializer Pass - Lowers package initializers to constant globals";
  }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final
  {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();

    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<int32_t> priorities;
    SmallVector<Attribute> symbols;

    // Walk all global operations in the module.
    module->walk(
      [&](GlobalOp globalOp)
      {
        if (!globalOp.getInitializerBlock() || succeeded(globalOp.hasValidInitializer()))
        {
          // Do not create a ctor function for this global.
          return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(globalOp);

        auto ctorSymbol = globalOp.getSymName().str() + "_ctor";
        auto fnT = mlir::go::FunctionType::get(builder.getContext(), {}, {});

        // Create a constructor function from this global.
        auto ctorFn = builder.create<mlir::go::FuncOp>(globalOp.getLoc(), ctorSymbol.c_str(), fnT);
        int priority = 0;
        if (auto priorityAttr = globalOp->getAttrOfType<IntegerAttr>("go.ctor.priority"))
        {
          priority = priorityAttr.getInt();
          ctorFn->setAttr("go.ctor.priority", priorityAttr);
        }

        // Insert into the ctors map.
        priorities.push_back(priority);
        symbols.push_back(FlatSymbolRefAttr::get(builder.getStringAttr(ctorSymbol)));

        // Copy blocks from the global over to the new function.
        IRMapping mapping;
        globalOp.getInitializerRegion().cloneInto(&ctorFn.getBody(), mapping);

        // Find and replace the yield operation.
        auto yieldOps = ctorFn.getBody().getOps<YieldOp>();
        if (!yieldOps.empty())
        {
          YieldOp yieldOp = *yieldOps.begin();
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(yieldOp);
          auto addrType =
            PointerType::get(&this->getContext(), yieldOp.getInitializerValue().getType());

          // Get the address of the global where the yielded value should be stored.
          Value addr =
            builder.create<AddressOfOp>(globalOp.getLoc(), addrType, globalOp.getSymName());

          // Store the yielded value at the global address.
          builder.create<StoreOp>(
            globalOp.getLoc(), yieldOp.getInitializerValue(), addr, UnitAttr(), UnitAttr());

          // Insert a void return.
          builder.create<mlir::go::ReturnOp>(globalOp->getLoc());

          // Remove the yield operation.
          yieldOp.erase();
        }

        // Clone the global operation without its original regions.
        auto newGlobal = globalOp.cloneWithoutRegions();
        builder.insert(newGlobal);

        // Zero initialize this global.
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.createBlock(&newGlobal.getInitializerRegion());
          Value zero = builder.create<ZeroOp>(newGlobal.getLoc(), newGlobal.getGlobalType());
          builder.create<YieldOp>(newGlobal.getLoc(), zero);
        }

        // Erase the original global.
        globalOp.erase();
      });

    // Add all package initializers to the global ctors.
    module.walk(
      [&](mlir::go::FuncOp funcOp)
      {
        if (funcOp->hasAttr("package_initializer"))
        {
          const auto priority = funcOp->getAttrOfType<IntegerAttr>("priority");
          symbols.push_back(FlatSymbolRefAttr::get(context, funcOp.getSymName()));
          priorities.push_back(packageInitBasePriority + priority.getInt());
        }
      });

    // Create the LLVM ctors operation.
    builder.create<GlobalCtorsOp>(
      builder.getUnknownLoc(), builder.getArrayAttr(symbols), builder.getI32ArrayAttr(priorities));
  }
}; // namespace mlir::go

std::unique_ptr<Pass> createGlobalInitializerPass()
{
  return std::make_unique<GlobalInitializerPass>();
}
} // namespace mlir::go
