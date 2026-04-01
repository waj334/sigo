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
#define GEN_PASS_DEF_VALUENORMALIZATIONFUNCPASS
#define GEN_PASS_DEF_VALUENORMALIZATIONGLOBALPASS
#include "Go/Transforms/Passes.h.inc"

template <typename OpT, typename BaseT>
struct ValueNormalizationPass : public BaseT
{
  using BaseT::BaseT;

  void runOnOperation() override
  {
    const auto parentOp = this->getOperation();
    auto module = this->getOperation()->template getParentOfType<mlir::ModuleOp>();

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
          module.template lookupSymbol<mlir::go::GlobalConstantOp>(*constantRefOp.getSymRef());
        if (!globalConstantOp)
        {
          constantRefOp->emitOpError()
            << "no global constant found with symbol \"" << *constantRefOp.getSymRef() << "\"";
          this->signalPassFailure();
          return mlir::WalkResult::interrupt();
        }

        mlir::Value replacementValue;

        if (globalConstantOp.getValue())
        {
          // The global constant has a direct value attribute. Create a new
          // constant operation with that value at the reference site.
          rewriter.setInsertionPoint(constantRefOp);
          auto newConstOp = mlir::go::ConstantOp::create(rewriter, 
            loc, constantRefOp.getType(), *globalConstantOp.getValue(), mlir::StringAttr());
          replacementValue = newConstOp.getResult();
        }
        else
        {
          // The global constant has a body region. Inline it at the reference
          // site.
          auto& constantRegion = globalConstantOp.getBody();

          mlir::IRMapping mapping;

          rewriter.setInsertionPoint(constantRefOp);
          for (auto& op : constantRegion.front())
          {
            if (auto yieldOp = dyn_cast<mlir::go::YieldOp>(op))
            {
              replacementValue = mapping.lookup(yieldOp.getOperand());
              continue;
            }

            const auto newOp = rewriter.clone(op, mapping);
            newOp->setLoc(loc);
          }
        }

        if (!replacementValue)
        {
          globalConstantOp->emitOpError() << "global does not yield a value";
          this->signalPassFailure();
          return mlir::WalkResult::interrupt();
        }

        // Replace the original constant with the resolved value.
        rewriter.replaceOp(constantRefOp, replacementValue);

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
};

struct ValueNormalizationFuncPass
  : public ValueNormalizationPass<
      mlir::go::FuncOp,
      impl::ValueNormalizationFuncPassBase<ValueNormalizationFuncPass>>
{
  using ValueNormalizationPass::ValueNormalizationPass;
};

struct ValueNormalizationGlobalPass
  : public ValueNormalizationPass<
      mlir::go::GlobalOp,
      impl::ValueNormalizationGlobalPassBase<ValueNormalizationGlobalPass>>
{
  using ValueNormalizationPass::ValueNormalizationPass;
};

} // namespace mlir::go
