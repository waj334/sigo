
#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

template<typename resultT, typename opT, typename adaptorT>
std::optional<resultT> getOrFold(opT op, adaptorT adaptor)
{
  resultT operand;
  if (const auto attr = adaptor.getOperand())
  {
    operand = mlir::dyn_cast_or_null<resultT>(attr);
  }
  else if (auto definingOp = op->getOperand().getDefiningOp())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(definingOp->fold(results)) || results.empty())
    {
      return std::nullopt;
    }
    operand = mlir::dyn_cast_or_null<resultT>(mlir::cast<mlir::Attribute>(results.front()));
  }
  else
  {
    return std::nullopt;
  }

  if (!operand)
  {
    return std::nullopt;
  }

  return operand;
}

OpFoldResult ComplementOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }

  const auto& operand = *result;
  const auto type = mlir::cast<mlir::IntegerType>(operand.getType());
  const auto value = operand.getValue();

  // Bitwise NOT: x ^ -1
  const auto allOnes = llvm::APInt::getAllOnes(type.getWidth());
  return mlir::IntegerAttr::get(type, value ^ allOnes);
}

mlir::OpFoldResult NegCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }

  const auto& operand = *result;

  // Negate real and imaginary parts.
  const auto fT = operand.getReal().getType();
  const auto real = -operand.getReal().getValue();
  const auto imag = -operand.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult NegFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }

  const auto& operand = *result;
  return FloatAttr::get(operand.getType(), -operand.getValue());
}

OpFoldResult NegIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }

  const auto& operand = *result;
  return IntegerAttr::get(operand.getType(), -operand.getValue());
}

OpFoldResult NotOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }

  const auto& operand = *result;
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