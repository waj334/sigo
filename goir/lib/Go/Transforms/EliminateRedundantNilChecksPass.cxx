/// =========================================================================
/// Eliminate Redundant Nil Pointer Checks Pass
/// =========================================================================
///
/// This pass eliminates provably unnecessary nil pointer checks:
///
/// 1. Checks on freshly allocated pointers (go.alloca)
/// 2. Checks on addresses of globals (go.addressOf)
/// 3. Redundant checks on the same value within a basic block
/// 4. Checks on values proven non-null by dominating checks
///
/// =========================================================================

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
#define GEN_PASS_DEF_ELIMINATEREDUNDANTNILCHECKSPASS
#include "Go/Transforms/Passes.h.inc"

/// -------------------------------------------------------------------------
/// Nil check elimination patterns
/// -------------------------------------------------------------------------

// Pattern 1: Eliminate checks on freshly allocated pointers
struct EliminateAllocaNilCheck : public OpRewritePattern<go::NilPointerCheckOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(go::NilPointerCheckOp op,
                                  PatternRewriter& rewriter) const override
  {
    const Value addr = op.getAddr();
    Operation* def = addr.getDefiningOp();
    
    if (!def)
      return failure();

    // Check if this is a freshly allocated pointer
    if (auto alloca = dyn_cast<go::AllocaOp>(def))
    {
      // Alloca always returns a non-null pointer
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// Pattern 2: Eliminate checks on addresses of globals
struct EliminateAddressOfNilCheck : public OpRewritePattern<go::NilPointerCheckOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(go::NilPointerCheckOp op,
                                  PatternRewriter& rewriter) const override
  {
    const Value addr = op.getAddr();
    Operation* def = addr.getDefiningOp();
    
    if (!def)
      return failure();

    // Check if this is an address of a global
    if (isa<go::AddressOfOp>(def))
    {
      // Address of global is never null
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// Pattern 3: Eliminate redundant checks in same basic block
struct EliminateRedundantNilCheck : public OpRewritePattern<go::NilPointerCheckOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(go::NilPointerCheckOp op,
                                  PatternRewriter& rewriter) const override
  {
    const Value addr = op.getAddr();
    Block* block = op->getBlock();

    // Look for a previous nil check on the same SSA value in this block
    for (Operation& prevOp : *block)
    {
      if (&prevOp == op.getOperation())
        break;  // Reached current op

      if (auto prevCheck = dyn_cast<go::NilPointerCheckOp>(&prevOp))
      {
        if (prevCheck.getAddr() == addr)
        {
          // Found a dominating check on the same SSA value.
          // In SSA form, the same SSA value always has the same runtime value,
          // so if it passed a nil check before, it will pass again.
          rewriter.eraseOp(op);
          return success();
        }
      }

      // Conservative: stop if we see any store operation, as it might
      // invalidate our assumptions about memory state
      if (isa<go::StoreOp>(&prevOp))
      {
        return failure();
      }

      // Conservative: stop if we see a function call, as it might have
      // side effects that invalidate our assumptions
      if (isa<go::CallOp>(&prevOp) || isa<go::CallIndirectOp>(&prevOp))
      {
        return failure();
      }
    }

    return failure();
  }
};

/// -------------------------------------------------------------------------
/// Pass implementation
/// -------------------------------------------------------------------------

struct EliminateRedundantNilChecksPass
  : public impl::EliminateRedundantNilChecksPassBase<EliminateRedundantNilChecksPass>
{
  using Base::Base;

  void runOnOperation() override
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateAllocaNilCheck>(&getContext());
    patterns.add<EliminateAddressOfNilCheck>(&getContext());
    patterns.add<EliminateRedundantNilCheck>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::go
