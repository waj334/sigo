#include <optional>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpDefinition.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
{
  // Constant op constant-folds to its value.
  if (const auto value = adaptor.getValue())
  {
    return *value;
  }

  if (adaptor.getBody().empty())
  {
    // References a global constant value.
    if (const auto symbol = adaptor.getSymRef())
    {
      auto moduleOp = this->getOperation()->getParentOfType<ModuleOp>();
      if (auto refOp = mlir::dyn_cast<GlobalConstantOp>(moduleOp.lookupSymbol(*symbol)))
      {
        if (refOp.getValue())
        {
          // Return the value of the global constant.
          return *refOp.getValue();
        }

        // Fold the operation that defined the yielded value.
        auto yieldValueOp = refOp.getBody().front().getTerminator()->getOperand(0).getDefiningOp();

        // Try to fold the operation.
        SmallVector<OpFoldResult, 4> foldResults;
        if (failed(yieldValueOp->fold(foldResults)))
        {
          return {};
        }
        return foldResults.front();
      }
    }
  }
  else
  {
    // Evaluate the constant expression in the body.
    auto& block = adaptor.getBody().front();
    const auto yieldOp = mlir::dyn_cast<YieldOp>(block.getTerminator());
    const auto valueOp = yieldOp->getOperand(0).getDefiningOp();

    // Try to fold the operation.
    SmallVector<OpFoldResult, 4> foldResults;
    if (failed(valueOp->fold(foldResults)))
    {
      return {};
    }

    return foldResults.front();
  }

  return {};
}

LogicalResult ConstantOp::verify()
{
  const bool hasValue = getValue().has_value();
  const bool hasRegion = !this->getBody().empty();
  const bool hasRef = this->getSymRef().has_value();

  if (hasValue && hasRegion)
  {
    return emitOpError("cannot have both a value and a body");
  }

  if (hasValue && hasRef)
  {
    return emitOpError("cannot have both a reference and a value");
  }

  if (hasRef && hasRegion)
  {
    return emitOpError("cannot have both a reference and a body");
  }

  if (hasRef)
  {
    auto moduleOp = this->getOperation()->getParentOfType<ModuleOp>();
    const auto op = moduleOp.lookupSymbol(this->getSymRefAttr());
    if (!op)
    {
      return emitOpError() << "operation with symbol " << this->getSymRefAttr() << " does not exist";
    }
    if (!mlir::isa<GlobalConstantOp>(op))
    {
      return emitOpError("reference must be to a global constant");
    }
  }

  if (hasRegion)
  {
    auto& block = getBody().front();
    if (!mlir::isa<YieldOp>(block.getTerminator()))
    {
      return emitOpError("region must end with go.yield");
    }

    for (auto& op : block.without_terminator())
    {
      if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
      {
        if (!iface.hasNoEffect())
        {
          return op.emitOpError(
            "operation in constant body has memory effects and is not constant-like");
        }
      }
    }
  }

  return success();
}

LogicalResult GlobalConstantOp::verify()
{
  const bool hasValue = getValue().has_value();
  const bool hasRegion = !this->getBody().empty();

  if (hasValue && hasRegion)
  {
    return emitOpError("must have either a 'value' attribute or a region, but not both");
  }

  if (hasRegion)
  {
    auto& block = getBody().front();
    if (!mlir::isa<YieldOp>(block.getTerminator()))
    {
      return emitOpError("region must end with go.yield");
    }

    for (auto& op : block.without_terminator())
    {
      if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
      {
        if (!iface.hasNoEffect())
        {
          return op.emitOpError(
            "operation in constant body has memory effects and is not constant-like");
        }
      }
    }
  }

  return success();
}

} // namespace mlir::go