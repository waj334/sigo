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

/// Trace a pointer value back through GEP chains to its origin.
/// Returns true if the origin is provably non-null.
static bool isProvablyNonNull(mlir::Value addr)
{
  while (auto* def = addr.getDefiningOp())
  {
    if (isa<go::AddressOfOp>(def) || isa<go::AllocaOp>(def))
      return true;

    // String/slice indexing runtime functions either panic or return a valid
    // pointer — they never return null.
    if (isa<go::StringAddrOp>(def) || isa<go::SliceAddrOp>(def))
      return true;

    if (auto gep = dyn_cast<go::GetElementPointerOp>(def))
    {
      addr = gep.getValue();
      continue;
    }

    return false;
  }
  return false;
}

// Pattern 1: Eliminate checks on provably non-null pointers
// (alloca, addressOf, or any GEP chain rooted at one of those)
struct EliminateNonNullProvenanceNilCheck : public OpRewritePattern<go::NilPointerCheckOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(go::NilPointerCheckOp op,
                                PatternRewriter& rewriter) const override
  {
    if (isProvablyNonNull(op.getAddr()))
    {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/// Walk through a GEP chain to find the root base pointer.
static mlir::Value getGepRootBase(mlir::Value addr)
{
  while (auto gep = dyn_cast_or_null<go::GetElementPointerOp>(addr.getDefiningOp()))
    addr = gep.getValue();
  return addr;
}

// Pattern 2: Eliminate redundant checks in same basic block
struct EliminateRedundantNilCheck : public OpRewritePattern<go::NilPointerCheckOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(go::NilPointerCheckOp op,
                                PatternRewriter& rewriter) const override
  {
    const Value addr = op.getAddr();
    Block* block = op->getBlock();

    // If addr is derived from a GEP chain, find the root base pointer.
    // A nil check on the root in a dominating position proves this non-null too.
    const Value rootBase = getGepRootBase(addr);

    // Look for a previous nil check on the same SSA value (or root base) in this block
    for (Operation& prevOp : *block)
    {
      if (&prevOp == op.getOperation())
        break;  // Reached current op

      if (auto prevCheck = dyn_cast<go::NilPointerCheckOp>(&prevOp))
      {
        if (prevCheck.getAddr() == addr || prevCheck.getAddr() == rootBase)
        {
          rewriter.eraseOp(op);
          return success();
        }
      }

      // Conservative: stop if we see any store operation, as it might
      // invalidate our assumptions about memory state
      if (isa<go::StoreOp>(&prevOp))
        return failure();

      // Conservative: stop if we see a function call, as it might have
      // side effects that invalidate our assumptions
      if (isa<go::CallOp>(&prevOp) || isa<go::CallIndirectOp>(&prevOp))
        return failure();
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
    patterns.add<EliminateNonNullProvenanceNilCheck>(&getContext());
    patterns.add<EliminateRedundantNilCheck>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::go
