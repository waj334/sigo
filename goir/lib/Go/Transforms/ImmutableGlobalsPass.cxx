#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringSet.h>

#include <mlir/IR/IRMapping.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/Util.h"

namespace mlir::go
{
#define GEN_PASS_DEF_IMMUTABLEGLOBALSPASS
#include "Go/Transforms/Passes.h.inc"

struct ImmutableGlobalsPass : impl::ImmutableGlobalsPassBase<ImmutableGlobalsPass>
{
  using ImmutableGlobalsPassBase::ImmutableGlobalsPassBase;

  void runOnOperation() final
  {
    auto module = getOperation();
    llvm::StringSet<> mutableGlobals;

    // ---- Phase 1: identify mutable globals ----
    //
    // A global is mutable if any AddressOfOp referencing it has the address
    // (or any pointer derived from it via GEP) used as the destination of a
    // store, OR escapes to a context we cannot prove is read-only.
    //
    // Loads are explicitly safe: reading the value at the global's address
    // does not mutate the global, regardless of what happens to the loaded
    // value. (E.g., `gpio.Otyper.StoreBits(0x1)` loads Otyper's pointer
    // value and stores through it -- the global Gpio itself is not mutated.)

    module.walk(
      [&](go::AddressOfOp addrOp)
      {
        StringRef sym = addrOp.getSymbol();
        if (mutableGlobals.contains(sym))
          return; // already known mutable; skip
        if (isAddressMutated(addrOp.getResult()))
          mutableGlobals.insert(sym);
      });

    // Mark immutable globals.
    module.walk(
      [&](go::GlobalOp globalOp)
      {
        if (const StringRef sym = globalOp.getSymName(); mutableGlobals.contains(sym))
          return; // already known mutable from address analysis

        if (initializerRequiresRuntimeEval(globalOp))
          return; // ctor will mutate this; not immutable

        globalOp->setAttr("go.immutable", UnitAttr::get(&getContext()));
      });

    // ---- Phase 2: fold load(addressOf(@const_global)) in immutable inits ----

    SmallVector<go::GlobalOp> immutableGlobals;
    module.walk(
      [&](go::GlobalOp op)
      {
        if (op->hasAttr("go.immutable") && op.getInitializerBlock())
          immutableGlobals.push_back(op);
      });

    bool changed = true;
    while (changed)
    {
      changed = false;
      for (go::GlobalOp globalOp : immutableGlobals)
      {
        if (foldLoadsInInitRegion(globalOp))
          changed = true;
      }
    }
  }

  /// Returns true if `addr` (or anything derived from it via GEP) is used
  /// as the destination of a store, OR if the address escapes to a use we
  /// cannot prove is non-mutating.
  ///
  /// Returns false only if every reachable use is provably read-only:
  /// LoadOp, NilPointerCheckOp, or GEP whose own uses are all read-only.
  static bool isAddressMutated(Value addr)
  {
    llvm::SmallPtrSet<Operation*, 16> visited;
    return isAddressMutatedImpl(addr, visited);
  }

  static bool isAddressMutatedImpl(Value addr, llvm::SmallPtrSetImpl<Operation*>& visited)
  {
    for (Operation* user : addr.getUsers())
    {
      if (!visited.insert(user).second)
        continue;

      // Load: reads value at addr; does not mutate storage at addr.
      if (isa<go::LoadOp>(user))
        continue;

      // Nil-check: does not mutate or escape addr.
      if (isa<go::NilPointerCheckOp>(user))
        continue;

      // Store: mutating iff addr is the destination operand.
      if (auto storeOp = dyn_cast<go::StoreOp>(user))
      {
        if (storeOp.getAddr() == addr)
          return true; // direct mutation: *addr = ...
        if (storeOp.getValue() == addr)
          return true; // escape: storing addr into another slot
        // Defensive -- shouldn't reach here for a 2-operand store.
        return true;
      }

      // GEP: derived pointer into a sub-element. Recurse on its uses.
      if (auto gepOp = dyn_cast<go::GetElementPointerOp>(user))
      {
        if (isAddressMutatedImpl(gepOp.getResult(), visited))
          return true;
        continue;
      }

      // Anything else (call, return, cast not modeled above) is a
      // conservative escape -- assume the address can be mutated.
      return true;
    }
    return false;
  }

  /// Folds `load(addressOf(@const_global))` patterns in the global's init
  /// region by inlining the referenced global's init body in place.
  static bool foldLoadsInInitRegion(go::GlobalOp globalOp)
  {
    Block* block = globalOp.getInitializerBlock();
    if (!block)
      return false;

    bool anyChanged = false;

    for (Operation& op : llvm::make_early_inc_range(*block))
    {
      auto load = dyn_cast<go::LoadOp>(&op);
      if (!load)
        continue;

      auto addrOf = load.getOperand().getDefiningOp<go::AddressOfOp>();
      if (!addrOf)
        continue;

      auto referenced =
        SymbolTable::lookupNearestSymbolFrom<go::GlobalOp>(load, addrOf.getSymbolAttr());
      if (!referenced)
        continue;

      if (!referenced->hasAttr("go.immutable"))
        continue;

      Block* refBlock = referenced.getInitializerBlock();
      if (!refBlock)
        continue;

      auto refYield = dyn_cast<go::YieldOp>(refBlock->getTerminator());
      if (!refYield)
        continue;

      OpBuilder builder(load);
      IRMapping mapping;
      for (Operation& srcOp : *refBlock)
      {
        if (isa<go::YieldOp>(srcOp))
          break;
        builder.clone(srcOp, mapping);
      }

      load.getResult().replaceAllUsesWith(mapping.lookup(refYield.getInitializerValue()));
      load.erase();

      if (addrOf->use_empty())
        addrOf.erase();

      anyChanged = true;
    }

    return anyChanged;
  }

  static bool initializerRequiresRuntimeEval(go::GlobalOp globalOp)
  {
    // Fast path: already proven immutable in a prior iteration.
    if (globalOp->hasAttr("go.immutable"))
      return false;

    Block* block = globalOp.getInitializerBlock();
    if (!block)
      return false;

    for (Operation& op : *block)
    {
      if (isa<go::YieldOp>(op))
        continue;

      // Function calls are always runtime.
      if (isa<CallOpInterface>(&op))
        return true;

      // Loads aren't pure, but a load of an immutable global's address
      // is effectively a compile-time constant once the loads are folded
      // (Phase 2). We treat such a load as non-blocking.
      if (auto load = dyn_cast<go::LoadOp>(&op))
      {
        auto addrOf = load.getOperand().getDefiningOp<go::AddressOfOp>();
        if (!addrOf)
          return true; // load of non-global; can't reason about it

        auto referenced = SymbolTable::lookupNearestSymbolFrom<go::GlobalOp>(
            load, addrOf.getSymbolAttr());
        if (!referenced)
          return true;

        // Only the attribute is consulted -- no recursion. The fixed-point
        // loop guarantees that referenced globals get marked before they
        // become useful here.
        if (!referenced->hasAttr("go.immutable"))
          return true;

        continue;
      }

      // Anything else that isn't pure requires runtime evaluation.
      if (!isPure(&op))
        return true;
    }
    return false;
  }
};

} // namespace mlir::go