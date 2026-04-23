#include <optional>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpDefinition.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
{
  auto normalizeIntAttr = [&](mlir::Attribute attr) -> mlir::Attribute
  {
    const auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr);
    if (!intAttr)
    {
      return attr;
    }
    const auto resultType = mlir::dyn_cast<mlir::go::IntegerType>(getType());
    if (!resultType)
    {
      return attr;
    }
    size_t targetWidth;
    if (const auto width = resultType.getWidth(); width.has_value())
    {
      targetWidth = *width;
    }
    else
    {
      if (!this->getOperation()->getParentOp())
      {
        return attr;
      }
      const auto dataLayout = mlir::DataLayout::closest(this->getOperation());
      targetWidth = dataLayout.getTypeSizeInBits(resultType);
    }
    auto value = intAttr.getValue();
    if (value.getBitWidth() != targetWidth)
    {
      value = value.zextOrTrunc(targetWidth);
    }
    return mlir::IntegerAttr::get(mlir::IntegerType::get(this->getContext(), targetWidth), value);
  };

  // Constant op constant-folds to its value.
  if (const auto value = adaptor.getValue())
  {
    return normalizeIntAttr(*value);
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
          return normalizeIntAttr(*refOp.getValue());
        }
        // Fold the operation that defined the yielded value.
        auto yieldValueOp = refOp.getBody().front().getTerminator()->getOperand(0).getDefiningOp();
        SmallVector<OpFoldResult, 4> foldResults;
        if (succeeded(yieldValueOp->fold(foldResults)) && !foldResults.empty())
        {
          if (const auto attr = mlir::dyn_cast_or_null<mlir::Attribute>(foldResults.front()))
          {
            return normalizeIntAttr(attr);
          }
        }
      }
    }
  }
  else
  {
    // Evaluate the constant expression in the body.
    auto& block = adaptor.getBody().front();
    const auto yieldOp = mlir::dyn_cast<YieldOp>(block.getTerminator());
    const auto valueOp = yieldOp->getOperand(0).getDefiningOp();
    SmallVector<OpFoldResult, 4> foldResults;
    if (succeeded(valueOp->fold(foldResults)) && !foldResults.empty())
    {
      if (const auto attr = mlir::dyn_cast_or_null<mlir::Attribute>(foldResults.front()))
      {
        return normalizeIntAttr(attr);
      }
    }
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
      return emitOpError() << "operation with symbol " << this->getSymRefAttr()
                           << " does not exist";
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

OpFoldResult LiteralOp::fold(FoldAdaptor adaptor)
{
  return getValue();
}

} // namespace mlir::go