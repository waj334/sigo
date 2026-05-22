#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
namespace transforms::core
{

SmallVector<Value> convertBlockArgs(
  ConversionPatternRewriter& rewriter,
  mlir::Location loc,
  mlir::ValueRange inputs,
  Block* dest)
{
  SmallVector<Value> result;
  result.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i)
  {
    Value operand = inputs[i];
    if (dest)
    {
      const Type blockArgType = dest->getArgument(i).getType();
      if (operand.getType() != blockArgType)
      {
        auto castOp = mlir::UnrealizedConversionCastOp::create(
          rewriter, loc, SmallVector<Type>{ blockArgType }, SmallVector<Value>{ operand });
        result.push_back(castOp.getResult(0));
        continue;
      }
    }
    result.push_back(operand);
  }

  return result;
}

struct AddCOpLowering : public OpConversionPattern<AddCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::complex::AddOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AddFOpLowering : public OpConversionPattern<AddFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AddIOpLowering : public OpConversionPattern<AddIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AndOpLowering : public OpConversionPattern<AndOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AndNotOpLowering : public OpConversionPattern<AndNotOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    AndNotOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    mlir::Location loc = op.getLoc();

    auto rhsValue = adaptor.getRhs();
    const auto rhsType = rhsValue.getType();
    rewriter.setInsertionPointAfter(op->getPrevNode());

    uint64_t allOnes;
    if (mlir::go::cast<IntegerType>(op.getRhs().getType()).isSigned())
    {
      allOnes = INT64_MAX;
    }
    else
    {
      allOnes = UINT64_MAX;
    }

    // Calculate the complement of the RHS by XOR'ing the RHS and all ones
    auto allOnesConstOp = mlir::arith::ConstantIntOp::create(rewriter, loc, rhsType, allOnes);
    auto xOrOp =
      mlir::arith::XOrIOp::create(rewriter, loc, rhsType, rhsValue, allOnesConstOp.getResult());

    // And the complement with the LHS
    auto andOp = mlir::arith::AndIOp::create(rewriter, loc, adaptor.getLhs(), xOrOp.getResult());

    rewriter.replaceOp(op, andOp);
    return success();
  }
};

struct BitcastOpLowering : public OpConversionPattern<BitcastOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    BitcastOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const Type operandType = mlir::go::baseType(op.getValue().getType());
    const auto resultType = typeConverter->convertType(op.getType());

    // If the converted operand type already matches the result type, the
    // bitcast is a no-op (e.g. two named types with the same base type).
    if (adaptor.getValue().getType() == resultType)
    {
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(operandType)
      .Case<mlir::go::IntegerType, mlir::FloatType>(
        [&](auto)
        {
          rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, resultType, adaptor.getValue());
          return success();
        })
      .Case(
        [&](mlir::ComplexType)
        {
          rewriter.replaceOpWithNewOp<complex::BitcastOp>(op, resultType, adaptor.getValue());
          return success();
        })
      .Default(
        [&](auto) -> mlir::LogicalResult
        { return rewriter.notifyMatchFailure(op, "incompatible bitcast operation"); });
  }
};

struct BrOpLowering : public OpConversionPattern<BranchOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    BranchOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(), adaptor.getDestOperands());
    return success();
  }
};

struct BuiltInCallOpLowering : public OpConversionPattern<BuiltInCallOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    BuiltInCallOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto resultType = typeConverter->convertType(op.getResult(0).getType());
    const auto callee = op.getCallee();
    if (callee == "complex")
    {
      rewriter.replaceOpWithNewOp<mlir::complex::CreateOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    }
    else if (callee == "imag")
    {
      rewriter.replaceOpWithNewOp<mlir::complex::ImOp>(op, resultType, adaptor.getOperands()[0]);
    }
    else if (callee == "real")
    {
      rewriter.replaceOpWithNewOp<mlir::complex::ReOp>(op, resultType, adaptor.getOperands()[0]);
    }
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<CallOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
    {
      return op->emitError("failed to convert result types");
    }

    // NOTE: Closure calls are lowered by CallPass.

    rewriter.replaceOpWithNewOp<func::CallOp>(
      op, *op.getCallee(), resultTypes, adaptor.getOperands());
    return success();
  }
};

struct ComplexOpLowering : public OpConversionPattern<ComplexOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    ComplexOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::complex::CreateOp>(
      op, resultType, adaptor.getReal(), adaptor.getImag());
    return success();
  }
};

struct CmpCOpLowering : public OpConversionPattern<CmpCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CmpCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    mlir::Location loc = op.getLoc();
    rewriter.setInsertionPointAfter(op->getPrevNode());

    // Get the real and imaginary parts
    auto realLhsOp = mlir::complex::ReOp::create(rewriter, loc, adaptor.getLhs());
    auto imagLhsOp = mlir::complex::ImOp::create(rewriter, loc, adaptor.getLhs());

    auto realRhsOp = mlir::complex::ReOp::create(rewriter, loc, adaptor.getRhs());
    auto imagRhsOp = mlir::complex::ImOp::create(rewriter, loc, adaptor.getRhs());

    auto pred = mlir::arith::CmpFPredicate::OEQ;
    if (op.getPredicate() == CmpFPredicate::ne)
    {
      pred = mlir::arith::CmpFPredicate::ONE;
    }

    // SPEC: Complex types are comparable. Two complex values u and v are equal if both real(u) ==
    // real(v) and
    //       imag(u) == imag(v).
    auto cmpRealOp = mlir::arith::CmpFOp::create(
      rewriter, loc, pred, realLhsOp.getResult(), realRhsOp.getResult());
    auto cmpImagOp = mlir::arith::CmpFOp::create(
      rewriter, loc, pred, imagLhsOp.getResult(), imagRhsOp.getResult());
    auto andOp =
      mlir::arith::AndIOp::create(rewriter, loc, cmpRealOp.getResult(), cmpImagOp.getResult());

    rewriter.replaceOp(op, andOp);
    return success();
  }
};

struct CmpFOpLowering : public OpConversionPattern<CmpFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CmpFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    mlir::arith::CmpFPredicate pred;
    switch (op.getPredicate())
    {
      case CmpFPredicate::eq:
        pred = mlir::arith::CmpFPredicate::OEQ;
        break;
      case CmpFPredicate::gt:
        pred = mlir::arith::CmpFPredicate::OGT;
        break;
      case CmpFPredicate::ge:
        pred = mlir::arith::CmpFPredicate::OGE;
        break;
      case CmpFPredicate::lt:
        pred = mlir::arith::CmpFPredicate::OLT;
        break;
      case CmpFPredicate::le:
        pred = mlir::arith::CmpFPredicate::OLE;
        break;
      case CmpFPredicate::ne:
        pred = mlir::arith::CmpFPredicate::ONE;
        break;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(op, pred, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CmpIOpLowering : public OpConversionPattern<CmpIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CmpIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    mlir::arith::CmpIPredicate pred;
    switch (op.getPredicate())
    {
      case CmpIPredicate::eq:
        pred = mlir::arith::CmpIPredicate::eq;
        break;
      case CmpIPredicate::ne:
        pred = mlir::arith::CmpIPredicate::ne;
        break;
      case CmpIPredicate::slt:
        pred = mlir::arith::CmpIPredicate::slt;
        break;
      case CmpIPredicate::sle:
        pred = mlir::arith::CmpIPredicate::sle;
        break;
      case CmpIPredicate::sgt:
        pred = mlir::arith::CmpIPredicate::sgt;
        break;
      case CmpIPredicate::sge:
        pred = mlir::arith::CmpIPredicate::sge;
        break;
      case CmpIPredicate::ult:
        pred = mlir::arith::CmpIPredicate::ult;
        break;
      case CmpIPredicate::ule:
        pred = mlir::arith::CmpIPredicate::ule;
        break;
      case CmpIPredicate::ugt:
        pred = mlir::arith::CmpIPredicate::ugt;
        break;
      case CmpIPredicate::uge:
        pred = mlir::arith::CmpIPredicate::uge;
        break;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, pred, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CondBrOpLowering : public OpConversionPattern<CondBranchOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    CondBranchOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    SmallVector<Value> trueOperands =
      convertBlockArgs(rewriter, loc, adaptor.getTrueDestOperands(), op.getTrueDest());
    SmallVector<Value> falseOperands =
      convertBlockArgs(rewriter, loc, adaptor.getFalseDestOperands(), op.getFalseDest());

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
      op, adaptor.getCondition(), op.getTrueDest(), trueOperands, op.getFalseDest(), falseOperands);
    return success();
  }
};

struct DivCOpLowering : public OpConversionPattern<DivCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::complex::DivOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct DivFOpLowering : public OpConversionPattern<DivFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct DivSIOpLowering : public OpConversionPattern<DivSIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivSIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct DivUIOpLowering : public OpConversionPattern<DivUIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivUIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct GlobalOpLowering : public OpConversionPattern<GlobalOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    GlobalOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    // Create a replacement global operation.
    auto newGlobalOp = GlobalOp::create(
      rewriter,
      op.getLoc(),
      adaptor.getGlobalTypeAttr(),
      adaptor.getSymNameAttr(),
      adaptor.getSectionAttr(),
      adaptor.getAlignmentAttr());
    newGlobalOp->setAttr("llvm.linkage", op->getAttr("llvm.linkage"));

    if (!op.getInitializer().empty()) {
      rewriter.inlineRegionBefore(
          op.getInitializer(),
          newGlobalOp.getInitializerRegion(),
          newGlobalOp.getInitializerRegion().end());
    }

    // Remove the original global operation.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ImagOpLowering : public OpConversionPattern<ImagOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ImagOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::complex::ImOp>(op, resultType, adaptor.getValue());
    return success();
  }
};

struct MulCOpLowering : public OpConversionPattern<MulCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::complex::MulOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct MulFOpLowering : public OpConversionPattern<MulFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct MulIOpLowering : public OpConversionPattern<MulIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct OrOpLowering : public OpConversionPattern<OrOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct RealOpLowering : public OpConversionPattern<RealOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RealOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::complex::ReOp>(op, resultType, adaptor.getValue());
    return success();
  }
};

struct RemFOpLowering : public OpConversionPattern<RemFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RemFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct RemSIOpLowering : public OpConversionPattern<RemSIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RemSIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct RemUIOpLowering : public OpConversionPattern<RemUIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RemUIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<ReturnOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    SmallVector<mlir::Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert return types");
    }
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, resultTypes, adaptor.getOperands());
    return success();
  }
};

// coerceShiftAmount ensures the shift amount (rhs) has the same integer type
// as the value being shifted (lhs/result). Go allows different types for the
// shift value and shift amount, but arith shift ops require matching types.
static Value
coerceShiftAmount(ConversionPatternRewriter& rewriter, Location loc, Value rhs, Type resultType)
{
  if (rhs.getType() == resultType)
    return rhs;

  const auto rhsWidth = mlir::cast<mlir::IntegerType>(rhs.getType()).getWidth();
  const auto resultWidth = mlir::cast<mlir::IntegerType>(resultType).getWidth();
  if (rhsWidth < resultWidth)
    return arith::ExtUIOp::create(rewriter, loc, resultType, rhs);
  return arith::TruncIOp::create(rewriter, loc, resultType, rhs);
}

struct ShlOpLowering : OpConversionPattern<ShlOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ShlOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const Type resultType = typeConverter->convertType(op.getType());
    Value rhs = coerceShiftAmount(rewriter, op.getLoc(), adaptor.getRhs(), resultType);
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, resultType, adaptor.getLhs(), rhs);
    return success();
  }
};

struct ShrSIOpLowering : OpConversionPattern<ShrSIOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ShrSIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const Type resultType = typeConverter->convertType(op.getType());
    Value rhs = coerceShiftAmount(rewriter, op.getLoc(), adaptor.getRhs(), resultType);
    rewriter.replaceOpWithNewOp<arith::ShRSIOp>(op, resultType, adaptor.getLhs(), rhs);
    return success();
  }
};

struct ShrUIOpLowering : OpConversionPattern<ShrUIOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ShrUIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const Type resultType = typeConverter->convertType(op.getType());
    Value rhs = coerceShiftAmount(rewriter, op.getLoc(), adaptor.getRhs(), resultType);
    rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, resultType, adaptor.getLhs(), rhs);
    return success();
  }
};

struct SubCOpLowering : public OpConversionPattern<SubCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::complex::SubOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SubFOpLowering : public OpConversionPattern<SubFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SubIOpLowering : public OpConversionPattern<SubIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct XorOpLowering : public OpConversionPattern<XorOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    ConstantOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto origResultType = op.getType();
    auto resultType = typeConverter->convertType(op.getType());
    const auto foldedValue = *op.getValue();

    return mlir::TypeSwitch<mlir::Type, LogicalResult>(go::underlyingType(origResultType))
      .Case(
        [&](ComplexType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<ComplexNumberAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected ComplexNumberAttr for complex type");
          }
          rewriter.replaceOpWithNewOp<mlir::complex::ConstantOp>(
            op, resultType, ArrayAttr::get(getContext(), { value.getReal(), value.getImag() }));
          return success();
        })
      .Case(
        [&](IntegerType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<IntegerAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected IntegerAttr for integer type");
          }
          const auto resultValue = rewriter.getIntegerAttr(resultType, value.getInt());
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Case(
        [&](FloatType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<FloatAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected FloatAttr for float type");
          }
          const auto resultValue = rewriter.getFloatAttr(resultType, value.getValue());
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Case(
        [&](BooleanType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<mlir::BoolAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected BoolAttr for boolean type");
          }
          const auto resultValue = rewriter.getIntegerAttr(resultType, value.getValue() ? 1 : 0);
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Default([&](mlir::Type type)
               { return op->emitOpError("unhandled constant result type ") << type; });
  }
};

struct FloatTruncateOpLowering : public OpConversionPattern<FloatTruncateOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    FloatTruncateOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<mlir::go::FuncOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    mlir::go::FuncOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();

    SmallVector<NamedAttribute> attrs;
    attrs.reserve(adaptor.getAttributes().size());
    for (auto attr : adaptor.getAttributes())
    {
      attrs.push_back(attr);
    }

    // Store the Go function type as a type attribute for generating metadata later.
    attrs.emplace_back(
      rewriter.getStringAttr("originalType"), mlir::TypeAttr::get(op.getFunctionType()));

    const auto fnT =
      mlir::cast<mlir::FunctionType>(this->typeConverter->convertType(adaptor.getFunctionType()));
    auto newOp = mlir::func::FuncOp::create(rewriter, loc, adaptor.getSymName(), fnT, attrs);

    // Move the function body to the new function.
    rewriter.inlineRegionBefore(op.getFunctionBody(), newOp.getBody(), newOp.end());

    // Convert the argument types.
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *this->getTypeConverter())))
    {
      return rewriter.notifyMatchFailure(op, "region types conversion failed");
    }

    // Finally, erase the old function.
    rewriter.eraseOp(op);
    return success();
  }
};

struct FloatExtendOpLowering : public OpConversionPattern<FloatExtendOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    FloatExtendOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct IntTruncateOpLowering : public OpConversionPattern<IntTruncateOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    IntTruncateOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct LiteralOpLowering : public OpConversionPattern<LiteralOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    LiteralOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto origResultType = op.getType();
    auto resultType = typeConverter->convertType(op.getType());
    const auto foldedValue = op.getValue();

    return mlir::TypeSwitch<mlir::Type, LogicalResult>(go::underlyingType(origResultType))
      .Case(
        [&](ComplexType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<ComplexNumberAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected ComplexNumberAttr for complex type");
          }
          rewriter.replaceOpWithNewOp<mlir::complex::ConstantOp>(
            op, resultType, ArrayAttr::get(getContext(), { value.getReal(), value.getImag() }));
          return success();
        })
      .Case(
        [&](IntegerType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<IntegerAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected IntegerAttr for integer type");
          }
          const auto resultValue = mlir::IntegerAttr::get(resultType, value.getValue());
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Case(
        [&](FloatType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<FloatAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected FloatAttr for float type");
          }
          const auto resultValue = rewriter.getFloatAttr(resultType, value.getValue());
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Case(
        [&](BooleanType) -> LogicalResult
        {
          const auto value = mlir::dyn_cast<mlir::BoolAttr>(foldedValue);
          if (!value)
          {
            return op->emitOpError("expected BoolAttr for boolean type");
          }
          const auto resultValue = rewriter.getIntegerAttr(resultType, value.getValue() ? 1 : 0);
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, resultValue);
          return success();
        })
      .Default([&](mlir::Type type)
               { return op->emitOpError("unhandled constant result type ") << type; });
  }
};

struct SignedExtendOpLowering : public OpConversionPattern<SignedExtendOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    SignedExtendOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct ZeroExtendOpLowering : public OpConversionPattern<ZeroExtendOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    ZeroExtendOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct FloatToUnsignedIntOpLowering : public OpConversionPattern<FloatToUnsignedIntOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    FloatToUnsignedIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct FloatToSignedIntOpLowering : public OpConversionPattern<FloatToSignedIntOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    FloatToSignedIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct UnsignedIntToFloatOpLowering : public OpConversionPattern<UnsignedIntToFloatOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    UnsignedIntToFloatOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct SignedIntToFloatOpLowering : public OpConversionPattern<SignedIntToFloatOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    SignedIntToFloatOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(
      op, typeConverter->convertType(op.getResult().getType()), adaptor.getValue());
    return success();
  }
};

struct ComplementOpLowering : public OpConversionPattern<ComplementOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    ComplementOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    int64_t allOnes;
    if (mlir::go::cast<IntegerType>(op.getOperand().getType()).isSigned())
    {
      allOnes = INT64_MAX;
    }
    else
    {
      allOnes = UINT64_MAX;
    }

    auto constOp = mlir::arith::ConstantOp::create(
      rewriter,
      op.getLoc(),
      mlir::IntegerAttr::get(typeConverter->convertType(op.getResult().getType()), allOnes));
    rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
      op,
      typeConverter->convertType(op.getResult().getType()),
      adaptor.getOperand(),
      constOp.getResult());
    return success();
  }
};

struct NegCOpLowering : public OpConversionPattern<NegCOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NegCOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::complex::NegOp>(op, adaptor.getOperand());
    return success();
  }
};

struct NegFOpLowering : public OpConversionPattern<NegFOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NegFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::arith::NegFOp>(op, adaptor.getOperand());
    return success();
  }
};

struct NegIOpLowering : public OpConversionPattern<NegIOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NegIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto constOp = mlir::arith::ConstantOp::create(
      rewriter,
      op.getLoc(),
      mlir::IntegerAttr::get(typeConverter->convertType(op.getResult().getType()), 0));
    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, constOp.getResult(), adaptor.getOperand());
    return success();
  }
};

struct NotOpLowering : public OpConversionPattern<NotOp>
{
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto constOp = mlir::arith::ConstantOp::create(
      rewriter,
      op.getLoc(),
      mlir::IntegerAttr::get(typeConverter->convertType(op.getResult().getType()), 1));
    rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
      op,
      typeConverter->convertType(op.getResult().getType()),
      adaptor.getOperand(),
      constOp.getResult());
    return success();
  }
};
} // namespace transforms::core

void populateGoToCoreConversionPatterns(
  mlir::MLIRContext* context,
  TypeConverter& converter,
  RewritePatternSet& patterns)
{
  // clang-format off
        patterns.add<
            transforms::core::AddCOpLowering,
            transforms::core::AddFOpLowering,
            transforms::core::AddIOpLowering,
            transforms::core::AndOpLowering,
            transforms::core::AndNotOpLowering,
            transforms::core::BitcastOpLowering,
            transforms::core::BrOpLowering,
            transforms::core::BuiltInCallOpLowering,
            transforms::core::CallOpLowering,
            transforms::core::ComplexOpLowering,
            transforms::core::CmpCOpLowering,
            transforms::core::CmpFOpLowering,
            transforms::core::CmpIOpLowering,
            transforms::core::ComplementOpLowering,
            transforms::core::CondBrOpLowering,
            transforms::core::ConstantOpLowering,
            transforms::core::DivCOpLowering,
            transforms::core::DivFOpLowering,
            transforms::core::DivSIOpLowering,
            transforms::core::DivUIOpLowering,
            transforms::core::FloatExtendOpLowering,
            transforms::core::FloatToSignedIntOpLowering,
            transforms::core::FloatToUnsignedIntOpLowering,
            transforms::core::FloatTruncateOpLowering,
            transforms::core::FuncOpLowering,
            transforms::core::GlobalOpLowering,
            transforms::core::ImagOpLowering,
            transforms::core::IntTruncateOpLowering,
            transforms::core::LiteralOpLowering,
            transforms::core::MulCOpLowering,
            transforms::core::MulFOpLowering,
            transforms::core::MulIOpLowering,
            transforms::core::NegCOpLowering,
            transforms::core::NegFOpLowering,
            transforms::core::NegIOpLowering,
            transforms::core::NotOpLowering,
            transforms::core::OrOpLowering,
            transforms::core::RealOpLowering,
            transforms::core::RemFOpLowering,
            transforms::core::RemSIOpLowering,
            transforms::core::RemUIOpLowering,
            transforms::core::ReturnOpLowering,
            transforms::core::ShlOpLowering,
            transforms::core::ShrSIOpLowering,
            transforms::core::ShrUIOpLowering,
            transforms::core::SignedExtendOpLowering,
            transforms::core::SignedIntToFloatOpLowering,
            transforms::core::SubCOpLowering,
            transforms::core::SubFOpLowering,
            transforms::core::SubIOpLowering,
            transforms::core::UnsignedIntToFloatOpLowering,
            transforms::core::XorOpLowering,
            transforms::core::ZeroExtendOpLowering
        >(converter, context);
        // clang-format off
    }
}
