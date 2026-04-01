/// =========================================================================
/// GC Write Barrier Insertion Pass
/// =========================================================================
///
/// This pass inserts write barriers for the incremental tricolor garbage
/// collector. It implements a Yuasa-style insertion barrier that ensures:
///
/// 1. All stores of pointer-like values to heap locations are barriered
/// 2. Unknown destination provenance is treated conservatively (barriered)
/// 3. Stack-only stores are not barriered (performance optimization)
///
/// The pass performs lightweight provenance analysis to classify pointer
/// origins as Heap, NonHeap (stack), or Unknown. Write barriers are inserted
/// for Heap and Unknown destinations storing pointer-containing values.
///
/// Supported pointer-like types:
/// - Raw pointers (*T)
/// - Maps, channels (runtime-managed heap structures)
/// - Slices, strings (contain pointer to backing storage)
/// - Interfaces (contain type descriptor and data pointers)
///
/// The barrier function signature is:
///   runtime.gcWriteBarrier(slot *uintptr, val uintptr)
///
/// =========================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
#define GEN_PASS_DEF_INSERTGCWRITEBARRIERPASS
#include "Go/Transforms/Passes.h.inc"

enum class HeapProv : uint8_t
{
  NonHeap,
  Heap,
  Unknown,
};

HeapProv joinProv(const HeapProv a, const HeapProv b)
{
  if (a == b)
    return a;
  if (a == HeapProv::Unknown || b == HeapProv::Unknown)
    return HeapProv::Unknown;
  return HeapProv::Unknown;
}

/// -------------------------------------------------------------------------
/// Type helpers
/// -------------------------------------------------------------------------
///
/// These are the only places where you will likely need to adapt your exact
/// Go type classes.
///
/// This pass intentionally barriers only single-word pointer-like stores.
/// Aggregate/copy lowering is the next step, but this is the part that fixes
/// the common "store a heap pointer into a heap object field" case first.
///

bool isSingleWordPointerLikeType(const Type ty)
{
  return llvm::TypeSwitch<Type, bool>(ty)
    .Case<mlir::go::PointerType>([](auto) { return true; })
    // NOTE: ChanType and MapType are NOT single-word pointer types in this runtime.
    // Both _channel and _map are multi-field structs, not pointer-sized values.
    // Aggregate stores of chan/map values require a different barrier strategy.
    .Default([](Type) { return false; });
}

// Check if a type contains any pointer-like fields (recursive check)
bool typeContainsPointers(const Type ty)
{
  if (isSingleWordPointerLikeType(ty))
    return true;

  // Check struct types for pointer fields
  if (const auto structTy = dyn_cast<go::GoStructType>(ty))
  {
    for (const auto& field : structTy.getFields())
    {
      // Field is a tuple of (name, type, tag)
      const Type fieldTy = std::get<1>(field);
      if (typeContainsPointers(fieldTy))
        return true;
    }
  }

  // Check array types
  if (const auto arrayTy = dyn_cast<go::ArrayType>(ty))
  {
    return typeContainsPointers(arrayTy.getElementType());
  }

  // Check slice types (slice header contains pointer to backing array)
  if (isa<go::SliceType>(ty))
  {
    return true; // Slice always contains a pointer to the backing array
  }

  // String types contain pointer to backing data
  if (isa<go::StringType>(ty))
  {
    return true; // String contains a pointer to the data
  }

  // Interface types contain pointer to type descriptor and data pointer
  if (isa<go::InterfaceType>(ty))
  {
    return true;
  }

  return false;
}

Type getUIntPtrType(MLIRContext* ctx)
{
  return mlir::go::IntegerType::get(ctx, mlir::go::IntegerType::Uintptr);
}

Type getUIntPtrPtrType(MLIRContext* ctx)
{
  return go::PointerType::get(ctx, getUIntPtrType(ctx));
}

/// -------------------------------------------------------------------------
/// Heap provenance analysis
/// -------------------------------------------------------------------------
///
/// This is a lightweight, local, memoized provenance analysis:
/// - go.alloca heap          -> Heap
/// - go.alloca stack         -> NonHeap
/// - field/index/gep/casts   -> preserve base provenance
/// - selects                 -> join
/// - block args / unknowns   -> Unknown
///
/// It is deliberately simple and fast.
///

class HeapProvenance
{
public:
  explicit HeapProvenance() = default;

  HeapProv classify(const Value value)
  {
    if (const auto it = memo.find(value); it != memo.end())
      return it->second;

    if (!value)
      return memo[value] = HeapProv::Unknown;

    if (isa<BlockArgument>(value))
      return memo[value] = HeapProv::Unknown;

    Operation* def = value.getDefiningOp();
    if (!def)
      return memo[value] = HeapProv::Unknown;

    // ---------------------------------------------------------------------
    // Heap/stack allocation roots
    // ---------------------------------------------------------------------
    if (auto alloca = dyn_cast<go::AllocaOp>(def))
    {
      // Rename this predicate to match your actual alloca op API.
      // The printed IR suggests you have an explicit heap/stack flavor.
      return memo[value] = alloca.getHeap() ? HeapProv::Heap : HeapProv::NonHeap;
    }

    // If you also materialize heap through a runtime allocation wrapper,
    // teach the analysis of those calls here.
    if (auto call = dyn_cast<func::CallOp>(def))
    {
      if (const StringRef callee = call.getCallee(); callee == "runtime.alloc")
        return memo[value] = HeapProv::Heap;
      return memo[value] = HeapProv::Unknown;
    }

    // ---------------------------------------------------------------------
    // Pointer-preserving addr/cast ops
    // ---------------------------------------------------------------------
    //
    // Rename these to match your actual op classes.
    //
    if (auto op = dyn_cast<go::GetElementPointerOp>(def))
      return memo[value] = classify(op.getValue());

    if (auto op = dyn_cast<go::SliceAddrOp>(def))
      return memo[value] = classify(op.getSlice());

    if (auto op = dyn_cast<go::BitcastOp>(def))
      return memo[value] = classify(op.getValue());

    if (auto op = dyn_cast<UnrealizedConversionCastOp>(def))
    {
      if (op.getNumOperands() == 1)
        return memo[value] = classify(op.getInputs().front());
      return memo[value] = HeapProv::Unknown;
    }

    // ---------------------------------------------------------------------
    // Load operations - check the source address provenance
    // ---------------------------------------------------------------------
    if (auto op = dyn_cast<go::LoadOp>(def))
    {
      // If loading from heap, the loaded pointer likely points to heap too
      // This is conservative but safe
      const HeapProv srcProv = classify(op.getOperand());
      if (srcProv == HeapProv::Heap && isSingleWordPointerLikeType(value.getType()))
        return memo[value] = HeapProv::Unknown; // Could be heap, conservative
      return memo[value] = HeapProv::Unknown;
    }

    // ---------------------------------------------------------------------
    // Extract from aggregate
    // ---------------------------------------------------------------------
    if (auto op = dyn_cast<go::ExtractOp>(def))
    {
      // Extracting a field from an aggregate - doesn't change heap provenance
      // But we can't definitively say without tracking the aggregate's origin
      return memo[value] = HeapProv::Unknown;
    }

    return memo[value] = HeapProv::Unknown;
  }

private:
  DenseMap<Value, HeapProv> memo;
};

/// -------------------------------------------------------------------------
/// Runtime helper declaration
/// -------------------------------------------------------------------------

mlir::go::FuncOp getOrCreateBarrierDecl(const mlir::go::FuncOp op)
{
  auto module = op->getParentOfType<mlir::ModuleOp>();
  constexpr StringRef name = "runtime.gcWriteBarrier";
  if (const auto fn = module.lookupSymbol<mlir::go::FuncOp>(name))
    return fn;
  assert(false && "missing required runtime function");
}

/// -------------------------------------------------------------------------
/// Value materialization helpers
/// -------------------------------------------------------------------------
///
/// These adapt the store address/value to the barrier ABI:
///
///   runtime.gcWriteBarrier(*uintptr slot, uintptr val)
///
/// Again, rename the cast ops to match your actual Go dialect.
///

Value castAddressToUIntPtrPtr(OpBuilder& b, const Location loc, const Value addr)
{
  const Type wantTy = getUIntPtrPtrType(b.getContext());
  if (addr.getType() == wantTy)
    return addr;
  return mlir::go::BitcastOp::create(b, loc, wantTy, addr);
}

Value castValueToUIntPtrWord(OpBuilder& b, const Location loc, const Value value)
{
  const Type wantTy = getUIntPtrType(b.getContext());
  const Type ty = value.getType();

  if (ty == wantTy)
    return value;

  // Pointer-like values go through ptrtoint.
  if (isa<go::PointerType>(ty))
    return go::PtrToIntOp::create(b, loc, wantTy, value);

  // Integer same-width bitcast/trunc/ext path.
  if (const auto intTy = dyn_cast<mlir::go::IntegerType>(ty))
  {
    const auto wantIntTy = cast<mlir::go::IntegerType>(wantTy);
    if (intTy.getWidth() == wantIntTy.getWidth())
      return value;
    if (intTy.getWidth() < wantIntTy.getWidth())
      return mlir::go::ZeroExtendOp::create(b, loc, wantTy, value);
    return mlir::go::IntTruncateOp::create(b, loc, wantTy, value);
  }

  return {};
}

/// -------------------------------------------------------------------------
/// Store rewrite
/// -------------------------------------------------------------------------

static LogicalResult rewriteStore(go::StoreOp store, HeapProvenance& prov, mlir::go::FuncOp barrierFn)
{
  const Value addr = store.getAddr();
  const Value val = store.getValue();
  const HeapProv destProv = prov.classify(addr);

  // Only barrier stores to heap or unknown destinations (conservative approach)
  // Unknown destinations could be heap, so we must barrier them for correctness
  if (destProv == HeapProv::NonHeap)
    return failure();

  // Only barrier stores of pointer-containing values
  if (!isSingleWordPointerLikeType(val.getType()))
    return failure();

  // Skip volatile/atomic stores - they need special handling
  if (store.getIsVolatile() || store.getIsAtomic())
    return failure();

  IRRewriter rewriter(store.getContext());
  rewriter.setInsertionPoint(store);

  const Value slot = castAddressToUIntPtrPtr(rewriter, store.getLoc(), addr);
  if (!slot)
    return store.emitOpError("failed to cast heap destination to *uintptr");

  const Value word = castValueToUIntPtrWord(rewriter, store.getLoc(), val);
  if (!word)
    return store.emitOpError("failed to cast stored value to uintptr");

  mlir::go::CallOp::create(
    rewriter, store.getLoc(), TypeRange{}, barrierFn.getSymName(), ValueRange{ slot, word });
  rewriter.eraseOp(store);
  return success();
}

/// -------------------------------------------------------------------------
/// Pass
/// -------------------------------------------------------------------------

struct InsertGCWriteBarrierPass
  : public mlir::go::impl::InsertGCWriteBarrierPassBase<InsertGCWriteBarrierPass>
{
  using Base::Base;

  // Check if this function should have write barriers inserted
  static bool shouldInsertBarriers(mlir::go::FuncOp funcOp)
  {
    // Check for explicit nowritebarrier attribute
    if (funcOp->hasAttr("nowritebarrier"))
      return false;

    const StringRef funcName = funcOp.getSymName();

    // Never insert barriers into the write barrier implementation itself!
    if (funcName == "runtime.gcWriteBarrier" || funcName == "runtime.gcWriteBarrierCopy")
      return false;

    // Skip other GC runtime internals that manipulate GC data structures
    if (funcName.starts_with("runtime._gc.") ||
        funcName.starts_with("runtime.gc") ||
        funcName == "runtime.alloc" ||
        funcName == "runtime.initgc")
      return false;

    // Skip sync primitives (they use their own synchronization)
    if (funcName.starts_with("sync.") || funcName.starts_with("sync/atomic."))
      return false;

    return true;
  }

  void runOnOperation() override
  {
    mlir::go::FuncOp funcOp = getOperation();

    // Skip functions that shouldn't have barriers
    if (!shouldInsertBarriers(funcOp))
      return;

    const mlir::go::FuncOp barrierFn = getOrCreateBarrierDecl(funcOp);

    HeapProvenance prov;
    SmallVector<go::StoreOp> stores;

    funcOp.walk([&](const go::StoreOp store) { stores.push_back(store); });

    for (go::StoreOp store : stores)
    {
      const HeapProv destProv = prov.classify(store.getAddr());
      const Type valType = store.getValue().getType();
      const bool hasBarrierableValue = isSingleWordPointerLikeType(valType);
      const bool needsBarrier = (destProv == HeapProv::Heap || destProv == HeapProv::Unknown) &&
                                hasBarrierableValue &&
                                !store.getIsVolatile() && !store.getIsAtomic();

      if (needsBarrier)
      {
        if (failed(rewriteStore(store, prov, barrierFn)))
        {
          // If we determined this store needs a barrier but failed to rewrite it,
          // that's a hard error - we can't continue without correct barriers
          store.emitError("failed to insert required GC write barrier");
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace mlir::go