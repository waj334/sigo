#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

template<typename resultT, typename opT, typename adaptorT>
std::optional<std::tuple<resultT, resultT>> getOrFold(opT op, adaptorT adaptor)
{
  resultT lhs;
  resultT rhs;

  const Value larg = op->getLhs();
  const Value rarg = op->getRhs();

  if (const auto attr = adaptor.getLhs())
  {
    lhs = mlir::dyn_cast_or_null<resultT>(attr);
  }
  else if (auto definingOp = larg.getDefiningOp())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(definingOp->fold(results)) || results.empty())
    {
      return std::nullopt;
    }
    lhs = mlir::dyn_cast_or_null<resultT>(mlir::cast<mlir::Attribute>(results.front()));
  }
  else if (const auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(larg))
  {
    // TODO: The originating operation of a value passed to a PHI node should be able to be
    //       determined by examining uses.
    /*
    const auto parentBlock = blockArg.getOwner();
    const auto argIndex = blockArg.getArgNumber();
    for (const auto& pred : parentBlock->getPredecessors())
    {
      auto terminator = pred->getTerminator();
      const auto operand = terminator->getOperand(argIndex);
    }
    */
    return std::nullopt;
  }
  else
  {
    return std::nullopt;
  }

  if (const auto attr = adaptor.getRhs())
  {
    rhs = mlir::dyn_cast_or_null<resultT>(attr);
  }
  else if (auto definingOp = op->getRhs().getDefiningOp())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(definingOp->fold(results)) || results.empty())
    {
      return std::nullopt;
    }
    rhs = mlir::dyn_cast_or_null<resultT>(mlir::cast<mlir::Attribute>(results.front()));
  }
  else
  {
    return std::nullopt;
  }

  if (!lhs || !rhs)
  {
    return std::nullopt;
  }

  return std::make_tuple(lhs, rhs);
}

mlir::OpFoldResult AddCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  const auto real = lhs.getReal().getValue() + rhs.getReal().getValue();
  const auto imag = lhs.getImag().getValue() + rhs.getImag().getValue();
  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult AddFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  return FloatAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
}

OpFoldResult AddIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval + rval);
}

OpFoldResult AddStrOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<StringAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  return StringAttr::get(this->getContext(), lhs.getValue() + rhs.getValue());
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval & rval);
}

OpFoldResult AndNotOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval & ~rval);
}

mlir::OpFoldResult CmpCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  bool value = false;
  switch (adaptor.getPredicate())
  {
    case CmpFPredicate::eq:
      value = lhs.getReal().getValue() == rhs.getReal().getValue() &&
        lhs.getImag().getValue() == rhs.getImag().getValue();
      break;
    case CmpFPredicate::ne:
      value = lhs.getReal().getValue() != rhs.getReal().getValue() ||
        lhs.getImag().getValue() != rhs.getImag().getValue();
      break;
    default:
      assert(false && "unsupported predicate");
      return {};
  }
  return mlir::BoolAttr::get(getContext(), value);
}

OpFoldResult CmpFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<BoolAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  bool value = false;
  switch (adaptor.getPredicate())
  {
    case CmpFPredicate::eq:
      value = lhs.getValue() == rhs.getValue();
      break;
    case CmpFPredicate::gt:
      value = lhs.getValue() > rhs.getValue();
      break;
    case CmpFPredicate::ge:
      value = lhs.getValue() >= rhs.getValue();
      break;
    case CmpFPredicate::lt:
      value = lhs.getValue() < rhs.getValue();
      break;
    case CmpFPredicate::le:
      value = lhs.getValue() <= rhs.getValue();
      break;
    case CmpFPredicate::ne:
      value = lhs.getValue() != rhs.getValue();
      break;
  }
  return mlir::BoolAttr::get(getContext(), value);
}

mlir::OpFoldResult CmpIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }

  bool value = false;
  switch (adaptor.getPredicate())
  {
    case CmpIPredicate::eq:
      value = lval == rval;
      break;
    case CmpIPredicate::ne:
      value = lval != rval;
      break;
    case CmpIPredicate::slt:
      value = lval.slt(rval);
      break;
    case CmpIPredicate::sle:
      value = lval.sle(rval);
      break;
    case CmpIPredicate::sgt:
      value = lval.sgt(rval);
      break;
    case CmpIPredicate::sge:
      value = lval.sge(rval);
      break;
    case CmpIPredicate::ult:
      value = lval.ult(rval);
      break;
    case CmpIPredicate::ule:
      value = lval.ule(rval);
      break;
    case CmpIPredicate::ugt:
      value = lval.ugt(rval);
      break;
    case CmpIPredicate::uge:
      value = lval.uge(rval);
      break;
  }

  return mlir::BoolAttr::get(getContext(), value);
}

OpFoldResult CmpStringOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<StringAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  bool value = false;
  switch (adaptor.getPredicate())
  {
    case CmpPredicate::eq:
      value = lhs.getValue() == rhs.getValue();
      break;
    case CmpPredicate::ne:
      value = lhs.getValue() != rhs.getValue();
      break;
  }
  return mlir::BoolAttr::get(getContext(), value);
}

mlir::OpFoldResult DivCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  const auto fT = lhs.getReal().getType();
  // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
  const auto denom = rhs.getReal().getValue() * rhs.getReal().getValue()
                   + rhs.getImag().getValue() * rhs.getImag().getValue();
  const auto real = (lhs.getReal().getValue() * rhs.getReal().getValue()
                   + lhs.getImag().getValue() * rhs.getImag().getValue()) / denom;
  const auto imag = (lhs.getImag().getValue() * rhs.getReal().getValue()
                   - lhs.getReal().getValue() * rhs.getImag().getValue()) / denom;
  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult DivFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  return FloatAttr::get(lhs.getType(), lhs.getValue() / rhs.getValue());
}

OpFoldResult DivSIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.sdiv(rval));
}

OpFoldResult DivUIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.udiv(rval));
}

mlir::OpFoldResult MulCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  const auto fT = lhs.getReal().getType();
  // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  const auto real = lhs.getReal().getValue() * rhs.getReal().getValue()
                  - lhs.getImag().getValue() * rhs.getImag().getValue();
  const auto imag = lhs.getReal().getValue() * rhs.getImag().getValue()
                  + lhs.getImag().getValue() * rhs.getReal().getValue();
  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult MulFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  return FloatAttr::get(lhs.getType(), lhs.getValue() * rhs.getValue());
}

OpFoldResult MulIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval * rval);
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval | rval);
}

OpFoldResult RemFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  return FloatAttr::get(lhs.getType(), lhs.getValue().remainder(rhs.getValue()));
}

OpFoldResult RemSIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.srem(rval));
}

OpFoldResult RemUIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.urem(rval));
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.shl(rval));
}

OpFoldResult ShrUIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.lshr(rval));
}

OpFoldResult ShrSIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  return IntegerAttr::get(lhs.getType(), lval.ashr(rval));
}

mlir::OpFoldResult SubCOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<ComplexNumberAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  const auto real = lhs.getReal().getValue() - rhs.getReal().getValue();
  const auto imag = lhs.getImag().getValue() - rhs.getImag().getValue();
  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult SubFOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<FloatAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  return FloatAttr::get(lhs.getType(), lhs.getValue() - rhs.getValue());
}

OpFoldResult SubIOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  // TODO: Need to check if lhs is unsigned.
  return IntegerAttr::get(lhs.getType(), lval - rval);
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor)
{
  const auto result = getOrFold<IntegerAttr>(this, adaptor);
  if (!result)
  {
    return {};
  }
  const auto& [lhs, rhs] = *result;
  auto lval = lhs.getValue();
  auto rval = rhs.getValue();
  if (lval.getBitWidth() != rval.getBitWidth())
  {
    rval = rval.zextOrTrunc(lval.getBitWidth());
  }
  // TODO: Need to check if lhs is unsigned.
  return IntegerAttr::get(lhs.getType(), lval ^ rval);
}

void AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAnd(argRanges));
}

void AndNotOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAnd(argRanges));
}

void DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferDivU(argRanges));
}

void DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferDivS(argRanges));
}

void MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferMul(argRanges));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferOr(argRanges));
}

void RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferRemS(argRanges));
}

void RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferRemU(argRanges));
}

void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferShl(argRanges));
}

void ShrUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferShrU(argRanges));
}

void ShrSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferShrS(argRanges));
}

void SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferSub(argRanges));
}

void XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferXor(argRanges));
}

::mlir::LogicalResult CmpCOp::verify()
{
  // Only == and != operators allowed for complex numbers as they are not ordered.
  if (this->getPredicate() != CmpFPredicate::eq && this->getPredicate() != CmpFPredicate::ne)
  {
    return emitOpError()
      << "only `==` and `!=` operators allowed for complex numbers as they are not ordered";
  }
  return success();
}

mlir::LogicalResult CmpInterfaceOp::verify()
{
  // At least one parameter needs to be an interface type.
  if (
    !mlir::go::isa<InterfaceType>(this->getLhs().getType()) &&
    !mlir::go::isa<InterfaceType>(this->getRhs().getType()))
  {
    return emitOpError() << "at least one operand MUST be an interface type";
  }

  return success();
}

} // namespace mlir::go
