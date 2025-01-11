#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

void InlineAsmOp::getEffects(
  ::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>&
    effects)
{
  effects.emplace_back(mlir::MemoryEffects::Write::get());
  effects.emplace_back(mlir::MemoryEffects::Read::get());
}

mlir::LogicalResult InlineAsmOp::verify()
{
  // There MUST be as constraints as there are input operands.
  if (this->getConstraints().size() < this->getOperandValues().size())
  {
    return this->emitOpError() << "there are fewer constraints than input operands";
  }
  return success();
}

} // namespace mlir::go