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

  // Verify that all blocks terminate with a branch operation branches to the exit block.
  auto blocks = mlir::SmallVector<mlir::Block*>(this->getCaseDests());
  if (this->getHasDefault())
  {
    blocks.push_back(this->getDefaultDest());
  }

  for (const auto caseDest : blocks)
  {
    // TODO: This will erroneously affect cases that terminate with `goto`. Fix that.
    if (auto terminator = mlir::dyn_cast_or_null<BranchOp>(caseDest->getTerminator());
        terminator && terminator.getDest() != this->getExitDest())
    {
      this->emitOpError() << "block should terminate with a branch to the exit block";
    }
  }

  return mlir::success();
}

} // namespace mlir::go