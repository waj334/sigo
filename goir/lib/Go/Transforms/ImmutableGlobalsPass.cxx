#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

#include <llvm/ADT/StringSet.h>

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

    // Collect globals that are stored to or whose address escapes.
    module.walk(
      [&](go::AddressOfOp addrOp)
      {
        StringRef sym = addrOp.getSymbol();
        for (auto* user : addrOp->getUsers())
        {
          if (isa<go::StoreOp>(user))
          {
            // Mutable if the global address is the store destination (direct write)
            // or the stored value (pointer escapes into another location).
            mutableGlobals.insert(sym);
          }
          else if (
            !isa<go::LoadOp>(user) && !isa<go::NilPointerCheckOp>(user) &&
            !isa<go::GetElementPointerOp>(user))
          {
            // Address escapes to an unknown use — conservatively mark mutable.
            mutableGlobals.insert(sym);
          }
        }
      });

    // For GEP chains rooted at addressOf, check if any GEP user is a store destination
    // or escapes to an unknown operation.
    module.walk(
      [&](go::GetElementPointerOp gepOp)
      {
        // Walk through the GEP chain to find the root base pointer.
        Value base = gepOp.getValue();
        while (auto parentGep = dyn_cast_or_null<go::GetElementPointerOp>(base.getDefiningOp()))
          base = parentGep.getValue();

        auto addrOp = dyn_cast_or_null<go::AddressOfOp>(base.getDefiningOp());
        if (!addrOp)
          return;

        for (auto* user : gepOp->getUsers())
        {
          if (isa<go::StoreOp>(user))
          {
            // Mutable if the GEP result is the store destination (direct write)
            // or the stored value (pointer escapes into another location).
            mutableGlobals.insert(addrOp.getSymbol());
          }
          else if (
            !isa<go::LoadOp>(user) && !isa<go::NilPointerCheckOp>(user) &&
            !isa<go::GetElementPointerOp>(user))
          {
            mutableGlobals.insert(addrOp.getSymbol());
          }
        }
      });

    // Mark globals that are never stored to as immutable.
    module.walk(
      [&](go::GlobalOp globalOp)
      {
        if (!mutableGlobals.contains(globalOp.getSymName()))
        {
          globalOp->setAttr("go.immutable", UnitAttr::get(&getContext()));
        }
      });
  }
};

} // namespace mlir::go
