
#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

OpFoldResult ComplementOp::fold(FoldAdaptor adaptor)
{
  auto operand = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getOperand());
  if (!operand)
  {
    return {};
  }

  auto type = mlir::cast<mlir::IntegerType>(operand.getType());
  auto value = operand.getValue();

  // Bitwise NOT: x ^ -1
  auto allOnes = llvm::APInt::getAllOnes(type.getWidth());
  auto result = value ^ allOnes;

  return mlir::IntegerAttr::get(type, result);
}

mlir::OpFoldResult NegCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto operand = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getOperand());
  if (!operand)
  {
    return {};
  }

  // Negate real and imaginary parts.
  const auto fT = operand.getReal().getType();
  auto real = -operand.getReal().getValue();
  auto imag = -operand.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult NegFOp::fold(FoldAdaptor adaptor)
{
  const auto operand = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getOperand());
  if (!operand)
  {
    return {};
  }

  return FloatAttr::get(operand.getType(), -operand.getValue());
}

OpFoldResult NegIOp::fold(FoldAdaptor adaptor)
{
  const auto operand = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getOperand());
  if (!operand)
  {
    return {};
  }

  return IntegerAttr::get(operand.getType(), -operand.getValue());
}

OpFoldResult NotOp::fold(FoldAdaptor adaptor)
{
  const auto operand = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getOperand());
  if (!operand)
  {
    return {};
  }

  return IntegerAttr::get(operand.getType(), !operand.getValue());
}

void ComplementOp::inferResultRanges(
  ArrayRef<ConstantIntRanges> argRanges,
  SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void NegIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void NotOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

::mlir::ParseResult RecvOp::parse(::mlir::OpAsmParser& p, ::mlir::OperationState& result)
{
  ::mlir::OpAsmParser::UnresolvedOperand operand;
  Type resultType;
  mlir::UnitAttr commaOk;

  if (p.parseOperand(operand))
  {
    return p.emitError(p.getCurrentLocation(), "operand expected to be `chan` type");
  }

  if (succeeded(p.parseOptionalKeyword("commaOk")))
  {
    commaOk = mlir::UnitAttr::get(p.getContext());
    result.addAttribute("commaOk", commaOk);
  }

  if (p.parseColon() || p.parseType(resultType))
  {
    return p.emitError(p.getCurrentLocation(), "error parsing result type");
  }

  auto operandType = p.getBuilder().getType<ChanType>(resultType, ChanDirection::SendRecv);
  if (p.resolveOperand(operand, operandType, result.operands))
  {
    operandType = p.getBuilder().getType<ChanType>(resultType, ChanDirection::RecvOnly);
    if (p.resolveOperand(operand, operandType, result.operands))
    {
      return p.emitError(p.getCurrentLocation(), "could not resolve chan type");
    }
  }

  result.addTypes(resultType);
  return success();
}

void RecvOp::print(::mlir::OpAsmPrinter& p)
{
  p << " ";
  p.printOperand(this->getOperand());
  if (getCommaOk())
  {
    p << " commaOk";
  }
  p << " : ";
  p.printType(this->getType());
}
} // namespace mlir::go