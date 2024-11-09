#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{
Value convertToInt(
  PatternRewriter& rewriter,
  const DataLayout& layout,
  const Value& input,
  const Location& loc)
{
  Value result = input;
  const Type inputType = input.getType();
  const auto preferredT = IntegerType::get(rewriter.getContext(), IntegerType::Signed);
  const auto uintT = IntegerType::get(rewriter.getContext(), IntegerType::Unsigned);
  const auto uintPtrT = IntegerType::get(rewriter.getContext(), IntegerType::Uintptr);

  // Convert the index integer value if it is NOT !go.i.
  if (inputType != preferredT)
  {
    if (inputType == uintT || inputType == uintPtrT)
    {
      // Bitcast these values to !go.i.
      result = rewriter.create<BitcastOp>(loc, preferredT, input);
    }
    else
    {
      const auto indexT = go::cast<IntegerType>(inputType);
      const auto indexBitWidth = layout.getTypeSizeInBits(preferredT);
      if (indexT.getWidth() == indexBitWidth)
      {
        result = rewriter.create<BitcastOp>(loc, preferredT, input);
      }
      else if (indexT.getWidth() > indexBitWidth)
      {
        result = rewriter.create<IntTruncateOp>(loc, preferredT, input);
      }
      else if (indexT.isSigned())
      {
        result = rewriter.create<SignedExtendOp>(loc, preferredT, input);
      }
      else
      {
        result = rewriter.create<ZeroExtendOp>(loc, preferredT, input);
      }
    }
  }
  return result;
}

void SliceOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

LogicalResult SliceOp::verify()
{
  // The result can only be a string ONLY if the input is a string.
  if (go::isa<StringType>(this->getType()) && !go::isa<StringType>(this->getInput().getType()))
  {
    return this->emitOpError() << "the result can only be a string ONLY if the input is a string";
  }
  return success();
}

} // namespace mlir::go
