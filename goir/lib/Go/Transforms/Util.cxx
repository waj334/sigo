#include "Go/Transforms/Util.h"

namespace mlir::go
{

bool isImmutableGlobal(go::GlobalOp globalOp)
{
  return globalOp->hasAttr("go.immutable");
}

bool canMaterializeAsGlobalInitializer(go::GlobalOp globalOp)
{
  if (!isImmutableGlobal(globalOp))
    return false;

  auto* block = globalOp.getInitializerBlock();
  if (!block)
    return false;

  auto yieldOp = mlir::dyn_cast<go::YieldOp>(block->getTerminator());
  if (!yieldOp)
    return false;

  llvm::DenseMap<mlir::Value, bool> memo;
  llvm::DenseSet<go::GlobalOp> activeGlobals;

  activeGlobals.insert(globalOp);
  return isConstantInitializerValue(yieldOp.getOperand(), memo, activeGlobals);
}

bool isConstantInitializerValue(
  mlir::Value value,
  llvm::DenseMap<mlir::Value, bool>& memo,
  llvm::DenseSet<go::GlobalOp>& activeGlobals)
{
  if (auto it = memo.find(value); it != memo.end())
    return it->second;

  auto* op = value.getDefiningOp();
  if (!op)
    return memo[value] = false;

  if (mlir::isa<go::ConstantOp>(op))
    return memo[value] = true;

  if (mlir::isa<go::LiteralOp>(op))
    return memo[value] = true;

  if (mlir::isa<go::ZeroOp>(op))
    return memo[value] = true;

  if (auto intToPtr = mlir::dyn_cast<go::IntToPtrOp>(op))
  {
    return memo[value] = isConstantInitializerValue(intToPtr.getOperand(), memo, activeGlobals);
  }

  if (auto bitcast = mlir::dyn_cast<go::BitcastOp>(op))
  {
    return memo[value] = isConstantInitializerValue(bitcast.getOperand(), memo, activeGlobals);
  }

  if (auto insert = mlir::dyn_cast<go::InsertOp>(op))
  {
    return memo[value] = isConstantInitializerValue(insert.getAggregate(), memo, activeGlobals) &&
      isConstantInitializerValue(insert.getValue(), memo, activeGlobals);
  }

  if (auto load = mlir::dyn_cast<go::LoadOp>(op))
  {
    auto addressOf = load.getOperand().template getDefiningOp<go::AddressOfOp>();
    if (!addressOf)
      return memo[value] = false;

    auto referenced =
      mlir::SymbolTable::lookupNearestSymbolFrom<go::GlobalOp>(op, addressOf.getSymbolAttr());

    if (!referenced)
      return memo[value] = false;

    if (!isImmutableGlobal(referenced))
      return memo[value] = false;

    if (!activeGlobals.insert(referenced).second)
      return memo[value] = false;

    bool ok = canMaterializeImmutableGlobalInitializer(referenced, memo, activeGlobals);

    activeGlobals.erase(referenced);
    return memo[value] = ok;
  }

  return memo[value] = false;
}

bool canMaterializeImmutableGlobalInitializer(
  go::GlobalOp globalOp,
  llvm::DenseMap<mlir::Value, bool>& memo,
  llvm::DenseSet<go::GlobalOp>& activeGlobals)
{
  if (!isImmutableGlobal(globalOp))
    return false;

  auto* block = globalOp.getInitializerBlock();
  if (!block)
    return false;

  auto yieldOp = mlir::dyn_cast<go::YieldOp>(block->getTerminator());
  if (!yieldOp)
    return false;

  return isConstantInitializerValue(yieldOp.getOperand(), memo, activeGlobals);
}

mlir::Value createGlobalString(
  mlir::Location loc,
  mlir::OpBuilder& builder,
  mlir::StringRef name,
  mlir::StringRef value,
  mlir::StringRef section,
  mlir::LLVM::Linkage linkage)
{
  assert(
    builder.getInsertionBlock() && builder.getInsertionBlock()->getParentOp() &&
    "expected builder to point to a block constrained in an op");
  auto module = builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(module && "builder points to an op outside of a module");

  // Create the global at the entry of the module.
  OpBuilder moduleBuilder(module.getBodyRegion(), builder.getListener());
  MLIRContext* ctx = builder.getContext();
  const auto type = LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 8), value.size());
  auto global = LLVM::GlobalOp::create(
    moduleBuilder,
    loc,
    type,
    /*isConstant=*/true,
    linkage,
    name,
    builder.getStringAttr(value),
    /*alignment=*/0);

  if (!section.empty())
  {
    global.setSection(section);
  }

  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  // Get the pointer to the first character in the global string.
  Value globalPtr = LLVM::AddressOfOp::create(builder, loc, ptrType, global.getSymNameAttr());
  return LLVM::GEPOp::create(
    builder, loc, ptrType, type, globalPtr, ArrayRef<mlir::LLVM::GEPArg>{ 0, 0 });
}

} // namespace mlir::go
