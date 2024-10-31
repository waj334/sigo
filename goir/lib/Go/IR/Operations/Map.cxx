#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{
LogicalResult MapUpdateOp::canonicalize(MapUpdateOp op, PatternRewriter& rewriter)
{
  const auto loc = op.getLoc();
  const auto keyPtrType = PointerType::get(rewriter.getContext(), { op.getKey().getType() });
  const auto elemPtrType = PointerType::get(rewriter.getContext(), { op.getValue().getType() });

  // Copy the key and element value onto the stack.
  Value keyPtr =
    rewriter.create<AllocaOp>(loc, keyPtrType, op.getKey().getType(), 1, UnitAttr(), StringAttr());
  Value elemPtr = rewriter.create<AllocaOp>(
    loc, elemPtrType, op.getValue().getType(), 1, UnitAttr(), StringAttr());

  rewriter.create<StoreOp>(loc, op.getKey(), keyPtr, UnitAttr(), UnitAttr());
  rewriter.create<StoreOp>(loc, op.getValue(), elemPtr, UnitAttr(), UnitAttr());

  // Lower to runtime call operation.
  const SmallVector<Type> results = {};
  const SmallVector<Value> args = { op.getMap(), keyPtr, elemPtr };
  rewriter.replaceOpWithNewOp<RuntimeCallOp>(
    op, results, formatPackageSymbol("runtime", "mapUpdate"), args);
  return success();
}

LogicalResult MapLookupOp::canonicalize(MapLookupOp op, PatternRewriter& rewriter)
{
  const auto loc = op.getLoc();
  const auto keyType = op.getKey().getType();
  const auto keyPtrType = PointerType::get(rewriter.getContext(), keyType);
  const auto ptrType = PointerType::get(rewriter.getContext(), std::nullopt);
  const auto boolType = BooleanType::get(rewriter.getContext());
  const auto resultType = op.getType(0);

  // Store the key on the stack.
  Value keyAddr = rewriter.create<AllocaOp>(loc, keyPtrType, keyType, 1, UnitAttr(), StringAttr());
  rewriter.create<StoreOp>(loc, op.getKey(), keyAddr, UnitAttr(), UnitAttr());

  // Lower to runtime call operation.
  const SmallVector<Type> results = { ptrType, boolType };
  const SmallVector<Value> args = { op.getMap(), keyAddr };
  auto lookupOp =
    rewriter.create<RuntimeCallOp>(loc, results, formatPackageSymbol("runtime", "mapLookup"), args);

  Value resultPtr = lookupOp->getResult(0);
  Value ok = lookupOp->getResult(1);

  auto opBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();

  // Split the current block so a conditional branch can be inserted to cover the map lookup
  // semantics.
  const auto continuationBlock = rewriter.splitBlock(opBlock, opPosition);
  auto trueBlock = rewriter.createBlock(continuationBlock);
  auto falseBlock = rewriter.createBlock(continuationBlock);

  // The return value will be passed to the continuation block via a block parameter.
  continuationBlock->addArgument(resultType, loc);

  // Build the true block.
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(trueBlock);

    // Load the value.
    Value result = rewriter.create<LoadOp>(loc, resultType, resultPtr, UnitAttr(), UnitAttr());

    // Branch to the continuation block.
    const SmallVector<Value> brArgs = { result };
    rewriter.create<BranchOp>(loc, brArgs, continuationBlock);
  }

  // Build the false block.
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(falseBlock);

    // Create the zero value of the result type.
    Value zeroValue = rewriter.create<ZeroOp>(loc, resultType);

    // Branch to the continuation block while passing the zero value.
    const SmallVector<Value> brArgs = { zeroValue };
    rewriter.create<BranchOp>(loc, brArgs, continuationBlock);
  }

  // Terminate the original block with the conditional branch.
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(opBlock);

    // Conditionally branch to the true or false block.
    rewriter.create<CondBranchOp>(
      loc, ok, SmallVector<Value>(), SmallVector<Value>(), trueBlock, falseBlock);
  }

  // Is the ok result present?
  if (lookupOp->getNumResults() == 2)
  {
    rewriter.replaceAllUsesWith(op->getResult(1), ok);
  }

  // Remove the original operation.
  rewriter.eraseOp(op);

  return success();
}
} // namespace mlir::go
