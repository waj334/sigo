#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

auto StringRangeOp::verifyBodyBlockArgs() -> mlir::LogicalResult
{
  const auto numBodyArgs = this->getBodyBlock()->getNumArguments();
  const auto keyType = mlir::go::IntegerType::get(
    this->getContext(), mlir::go::IntegerType::SignednessSemantics::Signed, std::nullopt);
  const auto elementType = mlir::go::IntegerType::get(
    this->getContext(), mlir::go::IntegerType::SignednessSemantics::Signed, 32);

  if (numBodyArgs != 2)
  {
    return this->emitOpError() << "expected 1 body block arguments, got " << numBodyArgs;
  }

  if (this->getBodyBlock()->getArgument(0).getType() != keyType)
  {
    return this->emitOpError() << "expected body block key argument of type " << keyType << ", got "
                               << this->getBodyBlock()->getArgument(0).getType();
  }

  if (this->getBodyBlock()->getArgument(1).getType() != elementType)
  {
    return this->emitOpError() << "expected body block element argument of type " << elementType
                               << ", but got " << this->getBodyBlock()->getArgument(1).getType();
  }

  return success();
}

} // namespace mlir::go