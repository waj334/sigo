#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{
::mlir::LogicalResult AllocaOp::verify()
{
  auto resultT = go::dyn_cast<PointerType>(this->getType());
  if (!resultT)
  {
    return this->emitOpError() << "the alloca operation must return a pointer type";
  }

  if (resultT.getElementType() && *resultT.getElementType() != this->getElement())
  {
    return this->emitOpError() << "the alloca operation must return either !go.ptr or !go.ptr<"
                               << this->getElement() << ">";
  }

  return success();
}

::mlir::LogicalResult StoreOp::verify()
{
  auto addrType = go::cast<PointerType>(this->getAddr().getType());
  if (!addrType)
  {
    return this->emitOpError() << "address type must be a pointer";
  }

  if (addrType.getElementType())
  {
    const auto elementType = *addrType.getElementType();
    const auto valueType = this->getValue().getType();
    if (!isCompatibleType(valueType, elementType))
    {
      return this->emitOpError() << "value type " << this->getValue().getType()
                               << " is incompatible with pointer type " << addrType;
    }
  }

  return success();
}

::mlir::LogicalResult GetElementPointerOp::verify()
{
  // No constant index should be negative.
  for (auto value : this->getConstIndices())
  {
    if (value < 0 && (value & kValueIndexMask) > this->getDynamicIndices().size())
    {
      return this->emitOpError() << "constant indices cannot be negative";
    }
  }
  return success();
}

::mlir::LogicalResult GlobalOp::verify()
{
  // Globals MUST specify a type.
  if (!this->getGlobalType())
  {
    return this->emitOpError() << "globals MUST specify a type";
  }
  return success();
}

::mlir::LogicalResult YieldOp::verify()
{
  if (mlir::isa<GlobalOp>(this->getOperation()->getParentOp()))
  {
    auto globalOp = mlir::dyn_cast<GlobalOp>(this->getOperation()->getParentOp());
    const auto expectedType = globalOp.getGlobalType();
    const auto actualType = this->getInitializerValue().getType();
    if (actualType != expectedType && !mlir::isa<UntypedType>(actualType))
    {
      return this->emitOpError() << "expected to yield value: " << expectedType
                                 << "\ngot:" << actualType;
    }
  }

  return success();
}

} // namespace mlir::go
