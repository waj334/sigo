#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Iterators.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"
#include <Go/IR/GoTypes.h>

namespace mlir::go
{

template <typename OpT>
struct ValueNormalizationPass final
  : public mlir::PassWrapper<ValueNormalizationPass<OpT>, mlir::OperationPass<OpT>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValueNormalizationPass<OpT>)
  void runOnOperation() override
  {
    const auto parentOp = this->getOperation();
    auto moduleOp = this->getOperation()->template getParentOfType<mlir::ModuleOp>();

    // Set up the rewriter so the operations can be replaced.
    mlir::IRRewriter rewriter(&this->getContext());

    // Inline global constant references before inferring untyped types.
    parentOp->walk(
      [&](mlir::go::ConstantOp constantRefOp)
      {
        const auto loc = constantRefOp->getLoc();
        if (!constantRefOp.getSymRef())
        {
          // Skip constants that don't have any symbol reference.
          return mlir::WalkResult::skip();
        }

        auto globalConstantOp =
          moduleOp.template lookupSymbol<mlir::go::GlobalConstantOp>(*constantRefOp.getSymRef());
        if (!globalConstantOp)
        {
          constantRefOp->emitOpError()
            << "no global constant found with symbol \"" << *constantRefOp.getSymRef() << "\"";
          this->signalPassFailure();
          return mlir::WalkResult::interrupt();
        }

        // Get the body of the constant expression that will be copied to the location of the
        // referring operation.
        auto& constantRegion = globalConstantOp.getBody();

        mlir::IRMapping mapping;
        mlir::Value yieldedValue;

        // Begin inserting new operations at the location of the reference.
        rewriter.setInsertionPoint(constantRefOp);
        for (auto& op : constantRegion.front())
        {
          if (auto yieldOp = dyn_cast<mlir::go::YieldOp>(op))
          {
            // Capture the yielded value.
            yieldedValue = mapping.lookup(yieldOp.getOperand());
            continue;
          }

          // Clone this operation into the current block.
          const auto newOp = rewriter.clone(op, mapping);

          // Update the location of this operation.
          newOp->setLoc(loc);
        }

        if (!yieldedValue)
        {
          globalConstantOp->emitOpError() << "global does not yield a value";
          this->signalPassFailure();
          return mlir::WalkResult::interrupt();
        }

        // Replace the original constant with the yielded value.
        rewriter.replaceOp(constantRefOp, yieldedValue);

        // Continue...
        return mlir::WalkResult::advance();
      });

    // Visit all operations in the function in reverse order.
    mlir::SmallVector<mlir::Operation*> erasures;
    parentOp->template walk<mlir::WalkOrder::PostOrder, ::mlir::ReverseIterator>(
      [&](mlir::Operation* op)
      {
        if (llvm::is_contained(erasures, op))
        {
          // Skip any operation that is marked for erasure.
          return;
        }

        auto resolver = mlir::dyn_cast<mlir::go::UntypedTypeResolverInterface>(op);
        if (!resolver)
        {
          return;
        }

        const auto resolvedOperandsTypes = resolver.resolveOperandTypes();

#if 0
        static std::mutex mutex;
        mutex.lock();
        op->dump();
        assert(op->getNumOperands() == resolvedOperandsTypes.size());
        mutex.unlock();
#else
        assert(op->getNumOperands() == resolvedOperandsTypes.size());
#endif

        for (const auto [operand, expectedType] :
             llvm::zip(op->getOperands(), resolvedOperandsTypes))
        {
          const auto actualType = operand.getType();

          if (!mlir::isa<mlir::go::UntypedType>(actualType) || actualType == expectedType)
          {
            continue;
          }

          const auto definingOp = operand.getDefiningOp();
          if (!definingOp)
          {
            op->emitOpError("untyped operand is a block argument, cannot resolve it");
            this->signalPassFailure();
            return;
          }

          // Determine the index of the result that must be replaced.
          const unsigned resultIndex = llvm::cast<OpResult>(operand).getResultNumber();
          const auto result =
            this->rewriteOp(rewriter, definingOp, resultIndex, expectedType, erasures);
          if (failed(result))
          {
            definingOp->emitOpError("failed to rewrite the operation");
            this->signalPassFailure();
            return;
          }
        }
      });

    // Finally, clean up operations marked for erasure.
    for (mlir::Operation* op : erasures)
    {
      if (op->use_empty())
      {
        op->erase();
      }
      else
      {
        op->emitOpError("still has uses after attempted rewrite, cannot erase");
        this->signalPassFailure();
        return;
      }
    }
  }

  mlir::FailureOr<mlir::Operation*> rewriteOp(
    mlir::IRRewriter& rewriter,
    mlir::Operation* originalOp,
    const size_t resultIndex,
    const mlir::Type resolvedType,
    mlir::SmallVector<mlir::Operation*>& erasures)
  {
    // Rewrite the original operation.
    OperationState state(originalOp->getLoc(), originalOp->getName());

    // Copy all existing operands and attributes.
    state.addOperands(originalOp->getOperands());
    state.addAttributes(originalOp->getAttrs());

    for (auto& region : originalOp->getRegions())
    {
      auto newRegion = state.addRegion();

      mlir::IRMapping mapping;
      region.cloneInto(newRegion, mapping);
    }

    // Copy all result types but replace the one at resultIndex.
    SmallVector<Type> newResultTypes(originalOp->getResultTypes());
    newResultTypes[resultIndex] = resolvedType;
    state.addTypes(newResultTypes);

    // Recreate the operation.
    rewriter.setInsertionPoint(originalOp);
    Operation* newOp = rewriter.create(state);

    // Replace all results.
    for (auto [oldRes, newRes] : llvm::zip(originalOp->getResults(), newOp->getResults()))
    {
      oldRes.replaceAllUsesWith(newRes);
    }

    auto resolver = mlir::dyn_cast<mlir::go::UntypedTypeResolverInterface>(newOp);
    if (!resolver)
    {
      // Drop the old operation.
      erasures.insert(erasures.begin(), originalOp);
      return newOp;
    }

    const auto resolvedOperandsTypes = resolver.resolveOperandTypes();
    const auto resolvedResultsTypes = resolver.resolveResultTypes();

    for (auto [operand, resolvedType] : llvm::zip(newOp->getOperands(), resolvedOperandsTypes))
    {
      if (!mlir::isa<mlir::go::UntypedType>(operand.getType()) || operand.getType() == resolvedType)
      {
        continue;
      }

      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
      {
        originalOp->emitOpError("untyped operand is a block argument, cannot resolve it");
        this->signalPassFailure();
        return mlir::failure();
      }

      const unsigned definingOpResultIndex = llvm::cast<OpResult>(operand).getResultNumber();
      const auto result =
        this->rewriteOp(rewriter, definingOp, definingOpResultIndex, resolvedType, erasures);
      if (failed(result))
      {
        definingOp->emitOpError("failed to rewrite the operation");
        this->signalPassFailure();
        return result;
      }
    }

    // Drop the old operation.
    erasures.insert(erasures.begin(), originalOp);
    return newOp;
  }

  StringRef getArgument() const override final { return "go-value-normalization-pass"; }
  StringRef getDescription() const override final
  {
    return "normalizes values at the function level";
  }
  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
  }
};

std::unique_ptr<mlir::Pass> createValueNormalizationFuncPass()
{
  return std::make_unique<ValueNormalizationPass<mlir::go::FuncOp>>();
}

std::unique_ptr<mlir::Pass> createValueNormalizationGlobalPass()
{
  return std::make_unique<ValueNormalizationPass<mlir::go::GlobalOp>>();
}

} // namespace mlir::go
