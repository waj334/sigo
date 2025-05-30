
#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

mlir::OpFoldResult AddCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto lhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  auto real = lhs.getReal().getValue() + rhs.getReal().getValue();
  auto imag = lhs.getImag().getValue() + rhs.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult AddFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return FloatAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
}

OpFoldResult AddIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
}

OpFoldResult AddStrOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<StringAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<StringAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return StringAttr::get(this->getContext(), lhs.getValue() + rhs.getValue());
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue() & rhs.getValue());
}

OpFoldResult AndNotOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue() & ~rhs.getValue());
}

mlir::OpFoldResult CmpCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto lhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  if (!lhs || !rhs)
  {
    return {};
  }

  bool result = false;
  switch (adaptor.getPredicate())
  {
    case CmpFPredicate::eq:
      result = lhs.getReal().getValue() == rhs.getReal().getValue() &&
        lhs.getImag().getValue() == rhs.getImag().getValue();
      break;
    case CmpFPredicate::ne:
      result = lhs.getReal().getValue() != rhs.getReal().getValue() ||
        lhs.getImag().getValue() != rhs.getImag().getValue();
      break;
    default:
      assert(false && "unsupported predicate");
      return {};
  }
  return mlir::BoolAttr::get(getContext(), result);
}

OpFoldResult CmpFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  bool result = false;
  switch (adaptor.getPredicate())
  {
    case CmpFPredicate::eq:
      result = lhs.getValue() == rhs.getValue();
      break;
    case CmpFPredicate::gt:
      result = lhs.getValue() > rhs.getValue();
      break;
    case CmpFPredicate::ge:
      result = lhs.getValue() >= rhs.getValue();
      break;
    case CmpFPredicate::lt:
      result = lhs.getValue() < rhs.getValue();
      break;
    case CmpFPredicate::le:
      result = lhs.getValue() <= rhs.getValue();
      break;
    case CmpFPredicate::ne:
      result = lhs.getValue() != rhs.getValue();
      break;
  }
  return mlir::BoolAttr::get(getContext(), result);
}

mlir::OpFoldResult CmpIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  const auto lhsVal = lhs.getValue();
  const auto rhsVal = rhs.getValue();

  bool result = false;
  switch (adaptor.getPredicate())
  {
    case CmpIPredicate::eq:
      result = lhsVal == rhsVal;
      break;
    case CmpIPredicate::ne:
      result = lhsVal != rhsVal;
      break;
    case CmpIPredicate::slt:
      result = lhsVal.slt(rhsVal);
      break;
    case CmpIPredicate::sle:
      result = lhsVal.sle(rhsVal);
      break;
    case CmpIPredicate::sgt:
      result = lhsVal.sgt(rhsVal);
      break;
    case CmpIPredicate::sge:
      result = lhsVal.sge(rhsVal);
      break;
    case CmpIPredicate::ult:
      result = lhsVal.ult(rhsVal);
      break;
    case CmpIPredicate::ule:
      result = lhsVal.ule(rhsVal);
      break;
    case CmpIPredicate::ugt:
      result = lhsVal.ugt(rhsVal);
      break;
    case CmpIPredicate::uge:
      result = lhsVal.uge(rhsVal);
      break;
  }

  return mlir::BoolAttr::get(getContext(), result);
}

OpFoldResult CmpStringOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<StringAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<StringAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  bool result = false;
  switch (adaptor.getPredicate())
  {
    case CmpPredicate::eq:
      result = lhs.getValue() == rhs.getValue();
      break;
    case CmpPredicate::ne:
      result = lhs.getValue() != rhs.getValue();
      break;
  }
  return mlir::BoolAttr::get(getContext(), result);
}

mlir::OpFoldResult DivCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto lhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  auto real = lhs.getReal().getValue() / rhs.getReal().getValue();
  auto imag = lhs.getImag().getValue() / rhs.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult DivFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return FloatAttr::get(lhs.getType(), lhs.getValue() / rhs.getValue());
}

OpFoldResult DivSIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().sdiv(rhs.getValue()));
}

OpFoldResult DivUIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().udiv(rhs.getValue()));
}

mlir::OpFoldResult MulCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto lhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  auto real = lhs.getReal().getValue() * rhs.getReal().getValue();
  auto imag = lhs.getImag().getValue() * rhs.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult MulFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return FloatAttr::get(lhs.getType(), lhs.getValue() * rhs.getValue());
}

OpFoldResult MulIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // TODO: Need to check if lhs is unsigned.
  return IntegerAttr::get(lhs.getType(), lhs.getValue().smul_sat(rhs.getValue()));
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor)
{
  mlir::IntegerAttr lhs;
  mlir::IntegerAttr rhs;

  if (!adaptor.getLhs())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(this->getLhs().getDefiningOp()->fold(results)))
    {
      return {};
    }
    lhs = mlir::dyn_cast_or_null<IntegerAttr>(results.front());
  }
  else
  {
    lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  }

  if (!adaptor.getRhs())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(this->getRhs().getDefiningOp()->fold(results)))
    {
      return {};
    }
    rhs = mlir::dyn_cast_or_null<IntegerAttr>(results.front());
  }
  else
  {
    rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  }

  if (!lhs || !rhs)
  {
    return {};
  }

  const auto i64Type = mlir::IntegerType::get(this->getContext(), 64);
  return IntegerAttr::get(i64Type, lhs.getValue() | rhs.getValue());
}

OpFoldResult RemFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return FloatAttr::get(lhs.getType(), lhs.getValue().remainder(rhs.getValue()));
}

OpFoldResult RemSIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhs.getValue()));
}

OpFoldResult RemUIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhs.getValue()));
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().shl(rhs.getValue()));
}

OpFoldResult ShrUIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().lshr(rhs.getValue()));
}

OpFoldResult ShrSIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().ashr(rhs.getValue()));
}

mlir::OpFoldResult SubCOp::fold(FoldAdaptor adaptor)
{
  // Both operands must be constants.
  auto lhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<ComplexNumberAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // Add real and imaginary parts.
  const auto fT = lhs.getReal().getType();
  auto real = lhs.getReal().getValue() - rhs.getReal().getValue();
  auto imag = lhs.getImag().getValue() - rhs.getImag().getValue();

  return ComplexNumberAttr::get(
    this->getContext(), FloatAttr::get(fT, real), FloatAttr::get(fT, imag));
}

OpFoldResult SubFOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  return FloatAttr::get(lhs.getType(), lhs.getValue() - rhs.getValue());
}

OpFoldResult SubIOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // TODO: Need to check if lhs is unsigned.
  return IntegerAttr::get(lhs.getType(), lhs.getValue() - rhs.getValue());
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor)
{
  const auto lhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  const auto rhs = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
  {
    return {};
  }

  // TODO: Need to check if lhs is unsigned.
  return IntegerAttr::get(lhs.getType(), lhs.getValue() ^ rhs.getValue());
}

void AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void AndNotOp::inferResultRanges(
  ArrayRef<ConstantIntRanges> argRanges,
  SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void ShrUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void ShrSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
}

void XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange)
{
  setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
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