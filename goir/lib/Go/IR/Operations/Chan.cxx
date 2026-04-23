#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

namespace mlir::go
{

auto ChanRangeOp::verifyBodyBlockArgs() -> mlir::LogicalResult
{
  const auto numBodyArgs = this->getBodyBlock()->getNumArguments();
  const auto chanType = mlir::go::dyn_cast<ChanType>(this->getChannel().getType());
  const auto elementType = chanType.getElementType();

  if (numBodyArgs == 0 || numBodyArgs > 2)
  {
    return this->emitOpError()
      << "body block must have exactly one argument or exactly two arguments";
  }

  if (const auto argType = this->getBodyBlock()->getArgument(0).getType(); argType != elementType)
  {
    return this->emitOpError() << "body block argument 0 should be of type " << elementType
                               << ", but got " << argType;
  }

  if (numBodyArgs == 2)
  {
    if (const auto argType = this->getBodyBlock()->getArgument(1).getType();
        mlir::go::isa<BooleanType>(argType))
    {
      return this->emitOpError() << "body block argument 1 should be of type !go.bool, but got "
                                 << argType;
    }
  }

  return mlir::success();
}

auto ChanSelectOp::verify() -> mlir::LogicalResult
{
  if (this->getChannel().size() != this->getSend().size())
  {
    return this->emitOpError() << "mismatch between channel size and send flags size";
  }

  if (this->getSend().size() != this->getCaseDests().size())
  {
    return this->emitOpError() << "mismatch between channel size and number of case blocks";
  }

  return mlir::success();
}

mlir::SmallVector<mlir::Type> ChanSendOp::resolveOperandTypes()
{
  mlir::SmallVector<mlir::Type> result(getOperation()->getOperandTypes());
  // The value operand (index 1) should match the channel's element type.
  if (auto chanType = mlir::dyn_cast<mlir::go::ChanType>(result[0]))
  {
    result[1] = chanType.getElementType();
  }
  return result;
}

} // namespace mlir::go