#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
#define GEN_PASS_DEF_FUNCPASS
#include "Go/Transforms/Passes.h.inc"

namespace
{
struct FuncPass : ::mlir::go::impl::FuncPassBase<FuncPass>
{
  using FuncPassBase<FuncPass>::FuncPassBase;

  void runOnOperation() override
  {
    const auto context = &this->getContext();

    OpBuilder builder(context);
    const auto i32Type = IntegerType::get(context, IntegerType::Signed, 32);
    const auto boolType = BooleanType::get(context);

    auto funcOp = getOperation();
    if (funcOp.getBody().empty())
    {
      return;
    }

    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto loc = funcOp->getLoc();
    auto* entryBlock = &*funcOp.getBody().begin();
    auto continueFromBlock = entryBlock;

    // Split the existing entry block so that stack allocations can be moved into it. This will also
    // be the block where defers will be initialized.
    auto entrySuccessor = entryBlock->splitBlock(entryBlock->begin());

    // Examine each alloca operation in this function.
    mlir::SmallVector<AllocaOp> allocaOpsToRelocate;
    funcOp.walk(
      [&](AllocaOp allocaOp)
      {
        if (!allocaOp.getHeapAttr())
        {
          // Move this allocation to the function's entry block.
          allocaOpsToRelocate.push_back(allocaOp);
        }
      });

    // Relocate stack allocations.
    for (const auto& allocaOp : allocaOpsToRelocate)
    {
      allocaOp->moveBefore(entryBlock, entryBlock->end());
    }

    bool hasDefer = false;
    funcOp.walk(
      [&](DeferOp)
      {
        hasDefer = true;
        return mlir::WalkResult::interrupt();
      });

    if (hasDefer)
    {
      builder.setInsertionPointToEnd(entryBlock);

      // Create the defer stack.
      const auto deferStackCreateFnType =
        mlir::dyn_cast<mlir::go::FuncOp>(moduleOp.lookupSymbol("runtime.deferStackCreate"))
          .getFunctionType();

      const auto deferStackType = deferStackCreateFnType.getResult(0);
      const auto jmpBufType = go::dyn_cast<GoStructType>(deferStackType).getFieldType(2);

      // Allocate memory for the defer stack.
      mlir::Value deferStackPtrValue = builder.create<AllocaOp>(
        loc,
        PointerType::get(context, deferStackType),
        deferStackType,
        1,
        mlir::UnitAttr(),
        mlir::StringAttr());

      // Designate this allocation as being the defer stack for this function. Defer call lowering
      // will look this allocation up later.
      deferStackPtrValue.getDefiningOp()->setAttr("deferStack", mlir::UnitAttr::get(context));

      // Create the defer stack via the respective runtime call.
      mlir::Value deferStackValue = builder
                                      .create<CallOp>(
                                        loc,
                                        SmallVector<mlir::Type>{ deferStackType },
                                        "runtime.deferStackCreate",
                                        mlir::ValueRange{})
                                      .getResult(0);

      // Store the defer stack value on the stack.
      builder.create<StoreOp>(
        loc, deferStackValue, deferStackPtrValue, mlir::UnitAttr(), mlir::UnitAttr());

      // The jmp environment from the defer stack.
      mlir::Value jmpEnvValue = builder.create<GetElementPointerOp>(
        loc,
        PointerType::get(context, jmpBufType),
        deferStackPtrValue,
        deferStackType,
        ValueRange{},
        mlir::SmallVector<int32_t>{ 0, 2 },
        mlir::SmallVector<bool>{false, false});

      // Set the jump point for defer stack unwinding.
      mlir::Value setJmpResult =
        builder
          .create<CallOp>(
            loc, SmallVector<mlir::Type>{ i32Type }, "setjmp", mlir::ValueRange{ jmpEnvValue })
          .getResult(0);

      // Initialize the defer stack, passing in the jump type value returned from setjmp.
      mlir::Value recoverResult = builder
                                    .create<CallOp>(
                                      loc,
                                      SmallVector<mlir::Type>{ boolType },
                                      "runtime.deferInit",
                                      mlir::ValueRange{ setJmpResult, deferStackPtrValue })
                                    .getResult(0);

      // Create the recover block at the end of this function. The recover block will just return
      // the zero value of this function's result type.
      auto recoveryBlock = new mlir::Block();
      recoveryBlock->insertAfter(&*funcOp.getBody().rbegin());
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(recoveryBlock);

        const auto fnType = funcOp.getFunctionType();
        const auto resultTypes = fnType.getResults();

        if (resultTypes.empty())
        {
          builder.create<ReturnOp>(loc);
        }
        else
        {
          mlir::SmallVector<Value> values;
          for (const auto& resultType : resultTypes)
          {
            values.push_back(builder.create<ZeroOp>(loc, resultType));
          }
          builder.create<ReturnOp>(loc, values);
        }
      }

      auto currentBlock = continueFromBlock;
      continueFromBlock = continueFromBlock->splitBlock(continueFromBlock->end());

      // Insert a conditional branch based on the recover result value. If it is true, then this
      // function should immediately return. Otherwise, this function executes normally.
      builder.setInsertionPointToEnd(currentBlock);
      builder.create<CondBranchOp>(
        loc, recoverResult, ValueRange{}, ValueRange{}, recoveryBlock, continueFromBlock);
    }

    // Finally, branch to the successor block.
    builder.setInsertionPointToEnd(continueFromBlock);
    builder.create<BranchOp>(loc, SmallVector<Value>{}, entrySuccessor);
  }
};

} // namespace
} // namespace mlir::go
