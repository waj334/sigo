#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

auto MapRangeOp::verifyBodyBlockArgs() -> mlir::LogicalResult
{
  const auto numBodyArgs = this->getBodyBlock()->getNumArguments();
  const auto mapType = mlir::go::dyn_cast<MapType>(this->getMap().getType());
  const auto keyType = mapType.getKeyType();
  const auto elementType = mapType.getValueType();

  if (numBodyArgs != 2)
  {
    return this->emitOpError() << "expected 2 body block arguments, got " << numBodyArgs;
  }

  if (this->getBodyBlock()->getArgument(0).getType() != keyType)
  {
    return this->emitOpError() << "expected key argument of type " << keyType << ", got "
                               << this->getBodyBlock()->getArgument(0).getType();
  }

  if (this->getBodyBlock()->getArgument(1).getType() != elementType)
  {
    return this->emitOpError() << "expected element argument of type " << elementType
                               << ", but got " << this->getBodyBlock()->getArgument(1).getType();
  }

  return success();
}

} // namespace mlir::go
