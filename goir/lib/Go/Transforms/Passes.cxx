#include "Go/Transforms/Passes.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{
struct LowerToCorePass : PassWrapper<LowerToCorePass, OperationPass<ModuleOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToCorePass)

  void runOnOperation() final
  {
    auto module = getOperation();

    CoreTypeConverter typeConverter(module);
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Target the module for rewriting
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<complex::ComplexDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    // populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    // populateCallOpTypeConversionPattern(patterns, typeConverter);
    // populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Mark illegal operations
    target.addIllegalOp<AddCOp>();
    target.addIllegalOp<AddFOp>();
    target.addIllegalOp<AddIOp>();
    target.addIllegalOp<AndOp>();
    target.addIllegalOp<AndNotOp>();

    target.addDynamicallyLegalOp<BitcastOp>(
      [](BitcastOp op)
      {
        if (const Type operandType = mlir::go::baseType(op.getValue().getType());
            mlir::go::isa<mlir::go::IntegerType>(operandType) ||
            mlir::go::isa<mlir::FloatType>(operandType) ||
            mlir::go::isa<mlir::ComplexType>(operandType))
        {
          return false;
        }
        return true;
      });

    target.addIllegalOp<BranchOp>();

    target.addDynamicallyLegalOp<BuiltInCallOp>(
      [](BuiltInCallOp op)
      {
        return StringSwitch<bool>(op.getCallee())
          .Case("complex", false)
          .Case("imag", false)
          .Case("real", false)
          .Default(true);
      });

    target.addIllegalOp<CallOp>();
    target.addIllegalOp<ComplexOp>();
    target.addIllegalOp<CmpCOp>();
    target.addIllegalOp<CmpFOp>();
    target.addIllegalOp<CmpIOp>();
    target.addIllegalOp<ComplementOp>();
    target.addIllegalOp<CondBranchOp>();

    target.addDynamicallyLegalOp<ConstantOp>(
      [](ConstantOp op)
      {
        auto resultType = op.getResult().getType();

        // String constants are lowered by the LLVM pass.
        return go::isa<StringType>(resultType);
      });

    target.addIllegalOp<DeclareTypeOp>();
    target.addIllegalOp<DivCOp>();
    target.addIllegalOp<DivFOp>();
    target.addIllegalOp<DivSIOp>();
    target.addIllegalOp<DivUIOp>();

    target.addDynamicallyLegalOp<GlobalOp>(
      [](GlobalOp op)
      {
        if (!op.getInitializerBlock())
        {
          // Globals with no initializer block will be converted in the LLVM pass.
          return true;
        }

        // Globals with invalid initializers will be converted to functions during the this pass.
        return succeeded(op.hasValidInitializer());
      });

    target.addIllegalOp<FloatExtendOp>();
    target.addIllegalOp<FloatToUnsignedIntOp>();
    target.addIllegalOp<FloatToSignedIntOp>();
    target.addIllegalOp<FloatTruncateOp>();
    target.addIllegalOp<mlir::go::FuncOp>();
    target.addIllegalOp<GoOp>();
    target.addIllegalOp<ImagOp>();
    target.addIllegalOp<IntTruncateOp>();
    target.addIllegalOp<MapUpdateOp>();
    target.addIllegalOp<MakeInterfaceOp>();
    target.addIllegalOp<MulCOp>();
    target.addIllegalOp<MulFOp>();
    target.addIllegalOp<MulIOp>();
    target.addIllegalOp<NegCOp>();
    target.addIllegalOp<NegFOp>();
    target.addIllegalOp<NegIOp>();
    target.addIllegalOp<NotOp>();
    target.addIllegalOp<OrOp>();
    target.addIllegalOp<RealOp>();
    target.addIllegalOp<RemFOp>();
    target.addIllegalOp<RemSIOp>();
    target.addIllegalOp<RemUIOp>();
    target.addIllegalOp<ReturnOp>();
    target.addIllegalOp<ShlOp>();
    target.addIllegalOp<ShrSIOp>();
    target.addIllegalOp<ShrUIOp>();
    target.addIllegalOp<SignedExtendOp>();
    target.addIllegalOp<SignedIntToFloatOp>();
    target.addIllegalOp<SubCOp>();
    target.addIllegalOp<SubFOp>();
    target.addIllegalOp<SubIOp>();
    target.addIllegalOp<UnsignedIntToFloatOp>();
    target.addIllegalOp<XorOp>();

    target.addDynamicallyLegalOp<YieldOp>(
      [](YieldOp op)
      {
        // Only convert a yield op if it belongs to a global with an invalid initializer block.
        if (auto globalOp = op->getParentOfType<GlobalOp>())
        {
          return succeeded(globalOp.hasValidInitializer());
        }
        // Usage of yield outside a global operation initializer is invalid and won't be converted
        // by this pass.
        return true;
      });

    target.addIllegalOp<ZeroExtendOp>();

    // Mark legal operations
    target.addLegalOp<AddressOfOp>();
    target.addLegalOp<AddStrOp>();
    target.addLegalOp<AllocaOp>();
    target.addLegalOp<AtomicAddIOp>();
    target.addLegalOp<AtomicCompareAndSwapIOp>();
    target.addLegalOp<AtomicSwapIOp>();
    target.addLegalOp<func::CallIndirectOp>();
    target.addLegalOp<CallIndirectOp>();
    target.addLegalOp<ChangeInterfaceOp>();
    target.addLegalOp<func::ConstantOp>();
    target.addLegalOp<DeferOp>();
    target.addLegalOp<FunctionToPointerOp>();
    target.addLegalOp<GetElementPointerOp>();
    target.addLegalOp<InterfaceCallOp>();
    target.addLegalOp<IntToPtrOp>();
    target.addLegalOp<LoadOp>();
    target.addLegalOp<MakeOp>();
    target.addLegalOp<MakeInterfaceOp>();
    target.addLegalOp<PanicOp>();
    target.addLegalOp<PointerToFunctionOp>();
    target.addLegalOp<PtrToIntOp>();
    target.addLegalOp<RecoverOp>();
    target.addLegalOp<RecvOp>();
    target.addLegalOp<RuntimeCallOp>();
    target.addLegalOp<SliceOp>();
    target.addLegalOp<StoreOp>();
    target.addLegalOp<ExtractOp>();
    target.addLegalOp<InsertOp>();
    target.addLegalOp<TypeInfoOp>();
    target.addLegalOp<ZeroOp>();

    populateGoToCoreConversionPatterns(module.getContext(), typeConverter, patterns);

    // Partially lower
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "lower-go-to-core"; }

  StringRef getDescription() const final { return "Lower Go IR to core dialects"; }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<complex::ComplexDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }
};

struct LowerToLLVMPass : PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

  void runOnOperation() final
  {
    // Get the module that will be lowered to LLVM
    auto module = getOperation();
    auto dataLayout = DataLayout(module);

    LowerToLLVMOptions options(&getContext(), dataLayout);
    if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout"));
        dataLayoutStr)
    {
      llvm::DataLayout llvmDataLayout(dataLayoutStr);
      options.dataLayout = llvmDataLayout;
    }

    LLVMTypeConverter typeConverter(module, options);
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Target the module for rewriting
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // The Go dialect is illegal
    target.addIllegalDialect<GoDialect>();

    // Add the required patterns already in MLIR
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateComplexToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    // Add each of the patterns required to lower the Go dialect operations to their respective
    // LLVM dialect operations.

    populateGoToLLVMConversionPatterns(typeConverter, patterns);

    // Completely lower
    if (failed(applyFullConversion(module, target, std::move(patterns))))
    {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "lower-go-to-llvm"; }

  StringRef getDescription() const final { return "Lower Go IR to LLVM IR"; }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<complex::ComplexDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }
};

std::unique_ptr<Pass> createLowerToCorePass()
{
  return std::make_unique<LowerToCorePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass()
{
  return std::make_unique<LowerToLLVMPass>();
}
} // namespace mlir::go
