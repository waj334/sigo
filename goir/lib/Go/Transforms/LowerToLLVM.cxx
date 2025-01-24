
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/DLTI/Traits.h>
#include <mlir/Dialect/LLVMIR/NVVMOpsEnums.h.inc>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeInfo.h"
#include "Go/Util.h"

namespace mlir::go
{

mlir::Value convert(
  mlir::OpBuilder& builder,
  const mlir::DataLayout& layout,
  const mlir::LLVMTypeConverter* converter,
  const mlir::Value X,
  const mlir::Type from,
  const mlir::Type to,
  const mlir::Location loc)
{
  const mlir::Type convertedTo = converter->convertType(to);
  mlir::Value result =
    mlir::TypeSwitch<mlir::Type, mlir::Value>(from)
      .Case(
        [&](mlir::go::IntegerType from) -> mlir::Value
        {
          const intptr_t fromWidth = layout.getTypeSizeInBits(from);
          return mlir::TypeSwitch<mlir::Type, mlir::Value>(to)
            .Case(
              [&](mlir::go::IntegerType to) -> mlir::Value
              {
                const intptr_t toWidth = layout.getTypeSizeInBits(to);
                if (fromWidth == toWidth)
                {
                  // No change
                  return X;
                }

                if (fromWidth > toWidth)
                {
                  // Truncate the additional bits.
                  return builder.create<mlir::LLVM::TruncOp>(loc, convertedTo, X);
                }

                if (from.getSignedness() == mlir::go::IntegerType::Signed)
                {
                  // Perform a signed-extend to preserve the sign bit. It doesn't
                  // matter when converting to an unsigned integer.
                  return builder.create<mlir::LLVM::SExtOp>(loc, convertedTo, X);
                }

                // Otherwise, perform a zero extend to keep the value
                // non-negative. It doesn't matter when converting to a signed
                // integer type.
                return builder.create<mlir::LLVM::ZExtOp>(loc, convertedTo, X);
              })
            .Default([&](mlir::Type) { return mlir::Value(); });
        })
      .Default([&](mlir::Type) { return mlir::Value(); });

  assert(result && "impossible conversion");
  return result;
}

mlir::Type lookUpRuntimeType(mlir::ModuleOp module, mlir::StringRef name)
{
  auto moduleOp = module.getOperation();
  assert(
    moduleOp->hasAttr("go.runtimeTypes") &&
    "module MUST have the go.runtimeTypes dictionary attribute");
  auto typeMap = ::mlir::dyn_cast<mlir::DictionaryAttr>(moduleOp->getAttr("go.runtimeTypes"));
  const auto result = typeMap.template getAs<::mlir::TypeAttr>(name);
  if (!result)
  {
    return {};
  }
  return result.getValue();
}

static mlir::SmallVector<Value> createRuntimeCall(
  mlir::PatternRewriter& rewriter,
  const mlir::Location location,
  const std::string& funcName,
  const mlir::LLVMTypeConverter* typeConverter,
  const mlir::ArrayRef<mlir::Value>& args)
{
  // Format the fully qualified function name
  const std::string qualifiedFuncName = formatPackageSymbol("runtime", funcName);
  const auto callee = FlatSymbolRefAttr::get(rewriter.getContext(), qualifiedFuncName);
  SmallVector<Value, 4> resultValues;

  // Look up the function in the module.
  auto module = rewriter.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
  auto funcOp = mlir::dyn_cast<mlir::FunctionOpInterface>(module.lookupSymbol(qualifiedFuncName));

  // Get the function type.
  mlir::LLVM::LLVMFunctionType funcType;
  size_t numResults = 0;
  if (mlir::isa<mlir::func::FuncOp>(funcOp))
  {
    const auto _funcType = mlir::dyn_cast<mlir::FunctionType>(funcOp.getFunctionType());
    numResults = _funcType.getNumResults();

    mlir::TypeConverter::SignatureConversion result(_funcType.getNumInputs());
    const auto convertedType =
      typeConverter->convertFunctionSignature(_funcType, false, false, result);
    funcType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(convertedType);
  }
  else
  {
    // This function has already been lowered.
    funcType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(funcOp.getFunctionType());
    const auto originalFuncType = mlir::dyn_cast<mlir::go::FunctionType>(
      funcOp->getAttrOfType<mlir::TypeAttr>("originalType").getValue());
    numResults = originalFuncType.getNumResults();
  }

  // Create the call.
  auto callOp = rewriter.create<LLVM::CallOp>(location, funcType, callee, args);
  callOp.getProperties().operandSegmentSizes = { { static_cast<int32_t>(args.size()), 0 } };
  callOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});

  // Handle the call results.
  if (numResults < 2)
  {
    // Return directly
    resultValues.append(callOp.result_begin(), callOp.result_end());
  }
  else
  {
    // Unpack result struct
    resultValues.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i)
    {
      resultValues.push_back(
        rewriter.create<mlir::LLVM::ExtractValueOp>(callOp.getLoc(), callOp->getResult(0), i));
    }
  }
  return resultValues;
}

// Locate or create the defer stack
Value locateOrCreateDeferStack(
  Operation* op,
  const mlir::LLVMTypeConverter* typeConverter,
  ConversionPatternRewriter& rewriter)
{
  const auto voidPtrType = rewriter.getType<mlir::LLVM::LLVMPointerType>();

  // Check if the defer stack already exists in the parent function
  auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
  const auto loc = parentFunc->getLoc();
  for (Block& block : parentFunc.getBody().getBlocks())
  {
    for (Operation& innerOp : block.getOperations())
    {
      if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(&innerOp))
      {
        if (allocaOp->hasAttrOfType<UnitAttr>("deferStack"))
        {
          return allocaOp.getResult();
        }
      }
    }
  }

  OpBuilder::InsertionGuard guard(rewriter);
  Block& entryBlock = *parentFunc.getBody().begin();

  auto newEntryBlock = rewriter.createBlock(&entryBlock);

  for (const auto& arg : entryBlock.getArguments())
  {
    newEntryBlock->addArgument(arg.getType(), arg.getLoc());
  }

  Value deferStackValue =
    createRuntimeCall(rewriter, loc, "deferStackCreate", typeConverter, {})[0];
  const auto deferStackType = deferStackValue.getType();

  Value sizeValue =
    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI32IntegerAttr(1));
  Value deferStackPtr =
    rewriter.create<LLVM::AllocaOp>(loc, voidPtrType, deferStackType, sizeValue);

  deferStackPtr.getDefiningOp()->setAttr("deferStack", rewriter.getUnitAttr());

  // Initialize the defer stack.
  rewriter.create<LLVM::StoreOp>(loc, deferStackValue, deferStackPtr);
  mlir::Value envPtr = rewriter.create<mlir::LLVM::GEPOp>(
    loc, voidPtrType, deferStackType, deferStackPtr, mlir::SmallVector<mlir::LLVM::GEPArg>{ 0, 2 });
  mlir::Value setjmpResult = rewriter
                               .create<mlir::LLVM::CallOp>(
                                 loc,
                                 mlir::SmallVector<mlir::Type>{ rewriter.getI32Type() },
                                 rewriter.getStringAttr("setjmp"),
                                 mlir::SmallVector<mlir::Value>{ envPtr })
                               .getResult();
  mlir::Value result = createRuntimeCall(
    rewriter, loc, "deferInit", typeConverter, { setjmpResult, deferStackPtr })[0];

  mlir::Block* recoverBlock;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    recoverBlock = rewriter.createBlock(&parentFunc.getBody(), parentFunc.getFunctionBody().end());

    if (const auto returnType = parentFunc.getFunctionType().getReturnType();
        !mlir::isa<mlir::LLVM::LLVMVoidType>(returnType))
    {
      // Return the zero value of the current function's result type.
      mlir::Value zeroValue = rewriter.create<mlir::LLVM::ZeroOp>(loc, returnType);
      rewriter.create<mlir::LLVM::ReturnOp>(loc, zeroValue);
    }
    else
    {
      rewriter.create<mlir::LLVM::ReturnOp>(loc, mlir::Value());
    }
  }

  // Insert a conditional branch after the init defer call to branch to the normal block if not
  // panicking or to the recover block upon recovering from a panic.
  rewriter.setInsertionPointAfter(result.getDefiningOp());
  rewriter.create<mlir::LLVM::CondBrOp>(
    loc, result, recoverBlock, &entryBlock, newEntryBlock->getArguments());

  return deferStackPtr;
}

namespace transforms::LLVM
{
struct AddressOfOpLowering : ConvertOpToLLVMPattern<AddressOfOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    AddressOfOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
      op, this->getVoidPtrType(), adaptor.getSymbol());
    return success();
  }
};

struct AddStrOpLowering : ConvertOpToLLVMPattern<AddStrOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(AddStrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();
    auto runtimeCallResults = createRuntimeCall(
      rewriter,
      loc,
      "stringConcat",
      this->getTypeConverter(),
      { adaptor.getLhs(), adaptor.getRhs() });
    rewriter.replaceOp(op, runtimeCallResults);
    return success();
  }
};

class AllocaOpLowering : public ConvertOpToLLVMPattern<AllocaOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(AllocaOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    OpBuilder::InsertionGuard guard(rewriter);
    const Location loc = op.getLoc();
    auto elementType = this->getTypeConverter()->convertType(adaptor.getElement());
    auto parentFunc = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    Value allocValue;

    if (adaptor.getHeap().has_value() && *adaptor.getHeap())
    {
      const auto module = op->getParentOfType<ModuleOp>();
      const DataLayout dataLayout(module);
      auto wordType =
        mlir::IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());

      // Get the size of the element type
      const auto allocationSize = dataLayout.getTypeSize(elementType);
      Value sizeValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, wordType, allocationSize * adaptor.getNumElements());

      // Create the runtime call to allocate memory on the heap
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "alloc", this->getTypeConverter(), { sizeValue });
      // Replace the original operation.
      rewriter.replaceOp(op, runtimeCallResults);
      allocValue = runtimeCallResults[0];
    }
    else
    {
      const auto funcLoc = parentFunc.getLoc();

      // Allocate the specified number of elements.
      Value sizeValue =
        rewriter
          .create<mlir::LLVM::ConstantOp>(funcLoc, rewriter.getI64Type(), adaptor.getNumElements())
          ->getResult(0);

      allocValue = rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), elementType, sizeValue);
      allocValue.getDefiningOp()->setLoc(funcLoc);

      // Move the constant operation before the alloca operation.
      sizeValue.getDefiningOp()->moveBefore(allocValue.getDefiningOp());

      // Zero initialize the value.
      Value zeroValue = rewriter.create<mlir::LLVM::ZeroOp>(funcLoc, elementType);
      rewriter.create<mlir::LLVM::StoreOp>(funcLoc, zeroValue, allocValue);

      auto newOp = allocValue.getDefiningOp();
      for (const auto attr : adaptor.getAttributes())
      {
        if (!newOp->hasAttr(attr.getName()))
        {
          newOp->setAttr(attr.getName(), attr.getValue());
        }
      }
    }

    // Create debug information if set on the operation.
    if (
      const auto fusedLoc =
        loc->findInstanceOf<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>())
    {
      rewriter.create<mlir::LLVM::DbgDeclareOp>(
        fusedLoc, allocValue, fusedLoc.getMetadata(), mlir::LLVM::DIExpressionAttr());
    }

    // return success.
    return success();
  }
};

struct AtomicAddIOpLowering : ConvertOpToLLVMPattern<AtomicAddIOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    AtomicAddIOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
      op,
      mlir::LLVM::AtomicBinOp::add,
      adaptor.getAddr(),
      adaptor.getRhs(),
      mlir::LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

struct AtomicCompareAndSwapIOpLowering : ConvertOpToLLVMPattern<AtomicCompareAndSwapIOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    AtomicCompareAndSwapIOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
      op.getLoc(),
      adaptor.getAddr(),
      adaptor.getOld(),
      adaptor.getValue(),
      mlir::LLVM::AtomicOrdering::seq_cst,
      mlir::LLVM::AtomicOrdering::seq_cst);

    // Extract the OK value from the result pair of the cmpxchg op
    Value ok = rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), cmpxchg, 1);

    rewriter.replaceOp(op, ok);
    return success();
  }
};

struct AtomicSwapIOpLowering : ConvertOpToLLVMPattern<AtomicSwapIOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    AtomicSwapIOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
      op,
      mlir::LLVM::AtomicBinOp::xchg,
      adaptor.getAddr(),
      adaptor.getRhs(),
      mlir::LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

struct BitcastOpLowering : ConvertOpToLLVMPattern<BitcastOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    BitcastOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto inputType = op.getValue().getType();
    const auto resultType = op.getType();

    const auto convertedInputType = this->getTypeConverter()->convertType(inputType);
    const auto convertedResultType = this->getTypeConverter()->convertType(resultType);
    Value inputValue = adaptor.getValue();

    if (
      mlir::go::baseType(inputType) == mlir::go::baseType(resultType) ||
      convertedInputType == convertedResultType)
    {
      // This is likely a type assertion.
      rewriter.replaceOp(op, inputValue);
      return success();
    }

    return mlir::TypeSwitch<Type, LogicalResult>(mlir::go::baseType(inputType))
      .Case<ChanType, InterfaceType, MapType, SliceType, StringType>(
        [&](auto t) -> LogicalResult
        {
          if (mlir::go::isa<GoStructType>(resultType))
          {
            rewriter.replaceOp(op, inputValue);
            return success();
          }
          return failure();
        })
      .Case(
        [&](FunctionType) -> LogicalResult
        {
          if (mlir::go::isa<PointerType>(resultType))
          {
            rewriter.replaceOp(op, inputValue);
            return success();
          }
          return failure();
        })
      .Case(
        [&](PointerType)
        {
          if (mlir::go::isa<FunctionType>(resultType))
          {
            rewriter.replaceOp(op, inputValue);
            return success();
          }

          if (mlir::go::isa<PointerType>(resultType))
          {
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, convertedResultType, inputValue);
            return success();
          }
          return failure();
        })
      .Default(
        [&](Type) -> LogicalResult
        {
          // This bitcast is not valid in the Go dialect.
          return failure();
        });
  }
};

struct BuiltInCallOpLowering : ConvertOpToLLVMPattern<BuiltInCallOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    BuiltInCallOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    const auto callee = op.getCallee();
    const ValueRange operands = adaptor.getOperands();
    const auto intType =
      mlir::IntegerType::get(this->getContext(), this->getTypeConverter()->getPointerBitwidth());
    const auto ptrType = mlir::LLVM::LLVMPointerType::get(this->getContext());
    const auto boolType = mlir::IntegerType::get(this->getContext(), 1);
    mlir::DataLayout dataLayout(module);

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
    {
      return failure();
    }

    if (callee == "append")
    {
      auto elementType = go::cast<SliceType>(op.getOperand(0).getType()).getElementType();
      auto elementTypeInfoGlobalOp = createTypeInfo(rewriter, module, loc, elementType);
      Value elementTypeInfoValue =
        rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
      const auto runtimeCallResults = createRuntimeCall(
        rewriter,
        loc,
        "sliceAppend",
        this->getTypeConverter(),
        { operands[0], operands[1], elementTypeInfoValue });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "cap")
    {
      const auto inputType = op.getOperandTypes()[0];
      llvm::TypeSwitch<Type>(inputType)
        .Case(
          [&](ArrayType type)
          { rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, intType, type.getLength()); })
        .Case(
          [&](ChanType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "channelCap", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "sliceCap", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "clear")
    {
      const auto inputType = op.getOperandTypes()[0];
      llvm::TypeSwitch<Type>(inputType)
        .Case(
          [&](MapType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "mapClear", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "sliceClear", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "close")
    {
      createRuntimeCall(rewriter, loc, "channelClose", this->getTypeConverter(), { operands[0] });
      rewriter.eraseOp(op);
    }
    else if (callee == "copy")
    {
      const auto srcType = op.getOperandTypes()[1];
      llvm::TypeSwitch<Type>(srcType)
        .Case(
          [&](SliceType type)
          {
            auto elementTypeInfoGlobalOp =
              createTypeInfo(rewriter, module, loc, type.getElementType());
            Value elementTypeInfoValue =
              rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
            const auto runtimeCallResults = createRuntimeCall(
              rewriter,
              loc,
              "sliceCopy",
              this->getTypeConverter(),
              { operands[0], operands[1], elementTypeInfoValue });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](StringType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter,
              loc,
              "sliceCopyString",
              this->getTypeConverter(),
              { operands[0], operands[1] });
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "delete")
    {
      // Store the key value on the stack.
      Value sizeValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, this->getIntPtrType(), 1);
      Value keyValue =
        rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, operands[1].getType(), sizeValue);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[1], keyValue);
      createRuntimeCall(
        rewriter, loc, "mapDelete", this->getTypeConverter(), { operands[0], keyValue });
      rewriter.eraseOp(op);
    }
    else if (callee == "len")
    {
      const auto inputType = op.getOperandTypes()[0];
      llvm::TypeSwitch<Type>(inputType)
        .Case(
          [&](ArrayType type)
          { rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, intType, type.getLength()); })
        .Case(
          [&](ChanType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "channelLen", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](MapType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "mapLen", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "sliceLen", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](StringType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "stringLen", this->getTypeConverter(), { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "make")
    {
      const auto resultType = op.getResultTypes()[0];
      TypeSwitch<Type>(resultType)
        .Case(
          [&](ChanType chanType)
          {
            SmallVector<Value> args;
            args.reserve(2);
            auto elementTypeInfoGlobalOp =
              createTypeInfo(rewriter, module, loc, chanType.getElementType());
            Value elementTypeInfoValue =
              rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
            args.push_back(elementTypeInfoValue);

            if (op.getNumOperands() == 0)
            {
              Value capacityValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, 0);
              args.push_back(capacityValue);
            }
            else
            {
              args.push_back(operands[0]);
            }

            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "channelMake", this->getTypeConverter(), args);
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](MapType mapType)
          {
            SmallVector<Value> args;
            args.reserve(3);
            auto keyTypeInfoGlobalOp = createTypeInfo(rewriter, module, loc, mapType.getKeyType());
            Value keyTypeInfoValue =
              rewriter.create<mlir::LLVM::AddressOfOp>(loc, keyTypeInfoGlobalOp);
            args.push_back(keyTypeInfoValue);

            auto elementTypeInfoGlobalOp =
              createTypeInfo(rewriter, module, loc, mapType.getValueType());
            Value elementTypeInfoValue =
              rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
            args.push_back(elementTypeInfoValue);

            if (op.getNumOperands() == 0)
            {
              Value capacityValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, 0);
              args.push_back(capacityValue);
            }
            else
            {
              args.push_back(operands[0]);
            }

            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "mapMake", this->getTypeConverter(), args);
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType sliceType)
          {
            SmallVector<Value> args;
            args.reserve(3);
            auto elementTypeInfoGlobalOp =
              createTypeInfo(rewriter, module, loc, sliceType.getElementType());
            Value elementTypeInfoValue =
              rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
            args.push_back(elementTypeInfoValue);
            args.push_back(operands[0]);

            if (op.getNumOperands() == 1)
            {
              args.push_back(operands[0]);
            }
            else
            {
              args.push_back(operands[1]);
            }

            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "sliceMake", this->getTypeConverter(), args);
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "max" || callee == "min")
    {
      const auto operandType = operands[0].getType();
      const auto originalOperandType = op.getOperand(0).getType();

      // TODO: Implement fast path for scenario where all values are constants.

      if (operands.size() == 1)
      {
        rewriter.replaceOp(op, { operands[0] });
      }
      else
      {
        const auto predecessorBlock = op->getBlock();

        // Create the exit block by splitting at the original operation.
        const auto exitBlock = rewriter.splitBlock(op->getBlock(), op->getIterator());
        const auto resultValue = exitBlock->addArgument(operandType, op->getResult(0).getLoc());
        rewriter.replaceOp(op, resultValue);

        // Create the initial block where the comparison will be performed.
        auto block = new mlir::Block;
        block->addArgument(operandType, operands[0].getLoc());
        block->insertBefore(exitBlock);

        // Branch to the first comparison block passing the first operand.
        rewriter.setInsertionPointToEnd(predecessorBlock);
        rewriter.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{ operands[0] }, block);

        // Populate each comparison block.
        for (size_t i = 1; i < operands.size(); ++i)
        {
          assert(block != exitBlock && "the compare block cannot be the exit block");
          const auto xValue = block->getArgument(0);
          const auto yValue = operands[i];

          // Set the insertion point to the start of the first comparison block.
          rewriter.setInsertionPointToStart(block);

          // Compare the incoming value against the next value using the respective comparison
          // operation.
          const Value cond = TypeSwitch<Type, Value>(originalOperandType)
                               .Case(
                                 [&](mlir::go::IntegerType) -> Value
                                 {
                                   mlir::LLVM::ICmpPredicate predicate;
                                   if (callee == "max")
                                   {
                                     predicate = isUnsigned(op->getOperandTypes()[0])
                                       ? mlir::LLVM::ICmpPredicate::ugt
                                       : mlir::LLVM::ICmpPredicate::sgt;
                                   }
                                   else
                                   {
                                     predicate = isUnsigned(op->getOperandTypes()[0])
                                       ? mlir::LLVM::ICmpPredicate::ult
                                       : mlir::LLVM::ICmpPredicate::slt;
                                   }
                                   return rewriter.create<mlir::LLVM::ICmpOp>(
                                     loc, boolType, predicate, xValue, yValue);
                                 })
                               .Case(
                                 [&](FloatType) -> Value
                                 {
                                   return rewriter.create<mlir::LLVM::FCmpOp>(
                                     loc,
                                     boolType,
                                     callee == "max" ? mlir::LLVM::FCmpPredicate::ogt
                                                     : mlir::LLVM::FCmpPredicate::olt,
                                     xValue,
                                     yValue);
                                 });

          mlir::Block* nextBlock = nullptr;
          if (i == operands.size() - 1)
          {
            // Branch to the exit block.
            nextBlock = exitBlock;
          }
          else
          {
            // Create a new block where the next comparison will be performed.
            nextBlock = new mlir::Block;
            block->addArgument(operandType, operands[i].getLoc());
            block->insertBefore(exitBlock);
            block = nextBlock;
          }

          // Branch to the next block passing the dependent value.
          rewriter.create<mlir::LLVM::CondBrOp>(
            loc,
            cond,
            nextBlock,
            SmallVector<Value>{ xValue },
            nextBlock,
            SmallVector<Value>{ yValue });
        }
      }
    }
    else if (callee == "new")
    {
      // This case should be handled by the heap escape pass.
      assert(false && "unreachable");
    }
    else if (callee == "panic")
    {
      createRuntimeCall(rewriter, loc, "_panic", this->getTypeConverter(), { operands[0] });
      rewriter.eraseOp(op);
    }
    else if (callee == "print")
    {
      createRuntimeCall(
        rewriter, loc, "_print", this->getTypeConverter(), SmallVector<Value>(operands));
      rewriter.eraseOp(op);
    }
    else if (callee == "println")
    {
      createRuntimeCall(
        rewriter, loc, "_println", this->getTypeConverter(), SmallVector<Value>(operands));
      rewriter.eraseOp(op);
    }
    else if (callee == "recover")
    {
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "_recover", this->getTypeConverter(), {});
      rewriter.replaceAllUsesWith(op.getResults(), runtimeCallResults);
    }
    else if (callee == "unsafe.Add")
    {
      Value addrValue =
        rewriter.create<mlir::LLVM::PtrToIntOp>(loc, this->getIntPtrType(), operands[0]);
      addrValue = rewriter.create<mlir::LLVM::AddOp>(loc, addrValue, operands[1]);
      rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, ptrType, addrValue);
    }
    else if (callee == "unsafe.Alignof")
    {
      const auto inputType = operands[0].getType();
      const auto value = dataLayout.getTypeABIAlignment(inputType);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, resultTypes[0], value);
    }
    else if (callee == "unsafe.Offsetof")
    {
      // TODO: The SSA generator needs to provide the indices into the struct.
    }
    else if (callee == "unsafe.Sizeof")
    {
      const auto inputType = operands[0].getType();
      const auto value = dataLayout.getTypeSize(inputType);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, resultTypes[0], value);
    }
    else if (callee == "unsafe.Slice")
    {
      const auto runtimeCallResults = createRuntimeCall(
        rewriter, loc, "slice", this->getTypeConverter(), { operands[0], operands[1] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.SliceData")
    {
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "sliceData", this->getTypeConverter(), { operands[0] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.String")
    {
      const auto runtimeCallResults = createRuntimeCall(
        rewriter, loc, "stringFromPointer", this->getTypeConverter(), { operands[0], operands[1] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.StringData")
    {
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "stringData", this->getTypeConverter(), { operands[0] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else
    {
      return failure();
    }
    return success();
  }
};

struct CallIndirectOpLowering : public ConvertOpToLLVMPattern<CallIndirectOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    CallIndirectOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), convertedResultTypes)))
    {
      return failure();
    }

    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands());

    operands.push_back(adaptor.getCallee());
    llvm::append_range(operands, adaptor.getCalleeOperands());

    auto callOp = rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), convertedResultTypes, operands);
    callOp.getProperties().operandSegmentSizes = { { static_cast<int32_t>(operands.size()), 0 } };
    callOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});

    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct ChangeInterfaceOpLowering : ConvertOpToLLVMPattern<ChangeInterfaceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ChangeInterfaceOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto resultType = this->getTypeConverter()->convertType(op.getType());

    // Create the type information for the interface's new type
    auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), op.getType());

    Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);

    // Alloca stack for the new value
    Value sizeValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
    Value addr =
      rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), resultType, sizeValue);

    // Get the underlying pointer value from the original interface
    Value ptrValue =
      rewriter.create<mlir::LLVM::ExtractValueOp>(loc, getVoidPtrType(), adaptor.getValue(), 1);

    // Store the pointer value in the new interface value
    Value newValue = rewriter.create<mlir::LLVM::UndefOp>(loc, resultType);
    newValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, newValue, infoValue, 0);
    newValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, newValue, ptrValue, 1);

    // Store the new type information
    rewriter.create<mlir::LLVM::StoreOp>(loc, newValue, addr);

    // Load the value
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, resultType, addr);
    return success();
  }
};

struct ChanRangeOpLowering : public ConvertOpToLLVMPattern<ChanRangeOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    ChanRangeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto ptrType = this->getVoidPtrType();
    const auto block = op->getBlock();
    const auto chanType = mlir::go::dyn_cast<ChanType>(op.getChannel().getType());
    const auto elementType = this->getTypeConverter()->convertType(chanType.getElementType());

    // Replace the range op with the respective runtime call.
    auto results = createRuntimeCall(
      rewriter, loc, "channelRange", this->getTypeConverter(), { adaptor.getChannel() });
    rewriter.eraseOp(op);

    mlir::Value addrVal = results[0];
    mlir::Value okValue = results[1];

    // Build the load block.
    mlir::Block* loadBlock;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      loadBlock = rewriter.createBlock(
        block->getParent(),
        std::next(block->getIterator()),
        mlir::SmallVector<mlir::Type>{ ptrType },
        mlir::SmallVector<mlir::Location>{ loc });
      Value value =
        rewriter.create<mlir::LLVM::LoadOp>(loc, elementType, loadBlock->getArgument(0));
      rewriter.create<mlir::LLVM::BrOp>(loc, SmallVector<mlir::Value>{ value }, op.getBodyBlock());
    }

    // Conditionally branch to the load block if the range iteration was successful. Otherwise,
    // branch to the exit block.
    rewriter.create<mlir::LLVM::CondBrOp>(
      loc,
      okValue,
      loadBlock,
      SmallVector<mlir::Value>{ addrVal },
      op.getExitBlock(),
      SmallVector<mlir::Value>{});

    return success();
  }
};

struct ChanSelectOpLowering : ConvertOpToLLVMPattern<ChanSelectOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ChanSelectOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    const auto chanType = this->getTypeConverter()->convertType(lookUpRuntimeType(module, "chan"));
    const auto ptrType = this->getVoidPtrType();
    const auto intType = this->getTypeConverter()->convertType(
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed));
    const auto boolType = rewriter.getI1Type();

    const mlir::Value zeroValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, 0);

    // Prepare input arrays.
    mlir::Value arrSize;
    mlir::Value chanArr;
    mlir::Value sendArr;
    mlir::Value readyArr;

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
      arrSize = rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, op.getChannel().size());
      chanArr = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, chanType, arrSize);
      sendArr = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, boolType, arrSize);
      readyArr = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, arrSize);
    }

    for (size_t i = 0; i < op.getChannel().size(); ++i)
    {
      const mlir::Value chanAddr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, chanType, chanArr, mlir::SmallVector<mlir::LLVM::GEPArg>{ i });
      rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getChannel()[i], chanAddr);

      const mlir::Value sendAddr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, boolType, sendArr, mlir::SmallVector<mlir::LLVM::GEPArg>{ i });
      const mlir::Value constValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, boolType, op.getSend()[i] ? 1 : 0);
      rewriter.create<mlir::LLVM::StoreOp>(loc, constValue, sendAddr);

      const mlir::Value readyAddr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, intType, readyArr, mlir::SmallVector<mlir::LLVM::GEPArg>{ i });
      rewriter.create<mlir::LLVM::StoreOp>(loc, zeroValue, readyAddr);
    }

    // Replace the operation with the channel select runtime call.
    const mlir::Value hasDefault =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, boolType, op.getHasDefault() ? 1 : 0);
    const auto result = createRuntimeCall(
      rewriter,
      loc,
      "channelSelect",
      this->getTypeConverter(),
      { chanArr, sendArr, readyArr, arrSize, hasDefault })[0];
    rewriter.eraseOp(op);

    // Create a compare block for each case.
    SmallVector<mlir::Block*> caseCmpBlocks;
    caseCmpBlocks.reserve(op.getCaseDests().size());
    for (const auto& caseDest : op.getCaseDests())
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto block = rewriter.createBlock(caseDest);
      caseCmpBlocks.push_back(block);
    }

    if (op.getHasDefault())
    {
      mlir::Block* defaultCmpBlock;
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block* successor = op.getExitDest();
        if (caseCmpBlocks.size() > 0)
        {
          successor = caseCmpBlocks[0];
        }

        // A negative result means branch to the default block.
        defaultCmpBlock =
          rewriter.createBlock(op->getParentRegion(), std::next(op->getBlock()->getIterator()));
        constexpr auto predicate = mlir::LLVM::ICmpPredicate::slt;
        const mlir::Value condition =
          rewriter.create<mlir::LLVM::ICmpOp>(loc, boolType, predicate, result, zeroValue);
        rewriter.create<mlir::LLVM::CondBrOp>(loc, condition, op.getDefaultDest(), successor);
      }
      rewriter.create<mlir::LLVM::BrOp>(loc, defaultCmpBlock);
    }
    else if (op.getCaseDests().size() > 0)
    {
      // Branch to the first case index compare block.
      rewriter.create<mlir::LLVM::BrOp>(loc, caseCmpBlocks[0]);
    }
    else
    {
      // This should be unreachable, but a terminator is required.
      rewriter.create<mlir::LLVM::BrOp>(loc, op.getExitDest());
    }

    for (size_t i = 0; i < op.getCaseDests().size(); i++)
    {
      mlir::Block* dest = op.getCaseDests()[i];
      mlir::Block* successor = op.getExitDest();
      if (i + 1 < caseCmpBlocks.size())
      {
        successor = caseCmpBlocks[i + 1];
      }

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(caseCmpBlocks[i]);
      const mlir::Value caseIndex = rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, i);
      constexpr auto predicate = mlir::LLVM::ICmpPredicate::eq;
      const mlir::Value condition =
        rewriter.create<mlir::LLVM::ICmpOp>(loc, boolType, predicate, result, caseIndex);
      rewriter.create<mlir::LLVM::CondBrOp>(loc, condition, dest, successor);
    }

    return success();
  }
};

struct ChanSendOpLowering : ConvertOpToLLVMPattern<ChanSendOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ChanSendOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto ptrType = this->getVoidPtrType();
    const auto elementType = adaptor.getValue().getType();

    // Copy the value into a new stack allocation.
    mlir::Value sendValuePtr;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
      const mlir::Value oneValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);

      // TODO: Eliminate this copy be requiring a pointer value be passed to the send operation.
      sendValuePtr = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, elementType, oneValue);
    }

    rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getValue(), sendValuePtr);

    // Replace the operation with the channel send runtime call.
    const auto result = createRuntimeCall(
      rewriter,
      loc,
      "channelSend",
      this->getTypeConverter(),
      { adaptor.getChannel(), sendValuePtr });
    rewriter.eraseOp(op);

    return success();
  }
};

struct ChanRecvOpLowering : ConvertOpToLLVMPattern<ChanRecvOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ChanRecvOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto chanType = mlir::cast<mlir::go::ChanType>(op.getChannel().getType());
    const auto elementType = this->getTypeConverter()->convertType(chanType.getElementType());
    const auto boolType = rewriter.getI1Type();

    const mlir::Value blockValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, boolType, op.getNumResults() == 2 ? 0 : 1);

    // Replace the operation with the channel receive runtime call.
    auto runtimeCallResults = createRuntimeCall(
      rewriter,
      loc,
      "channelReceive",
      this->getTypeConverter(),
      { adaptor.getChannel(), blockValue });

    // Load the value from the address returned by the runtime call.
    runtimeCallResults[0] =
      rewriter.create<mlir::LLVM::LoadOp>(loc, elementType, runtimeCallResults[0]);

    SmallVector<mlir::Value, 4> results;
    for (size_t i = 0; i < op.getNumResults(); i++)
    {
      results.push_back(runtimeCallResults[i]);
    }

    rewriter.replaceOp(op, results);

    return success();
  }
};

struct ConstantOpLowering : ConvertOpToLLVMPattern<ConstantOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ConstantOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    auto resultType = this->getTypeConverter()->convertType(op.getType());

    if (go::isa<StringType>(op.getType()))
    {
      const auto strAttr = mlir::dyn_cast<StringAttr>(op.getValue());
      const auto strLen = strAttr.size();
      const auto strHash = hash_value(strAttr.strref());
      const std::string name = "cstr_" + std::to_string(strHash);

      auto pointerT = this->getVoidPtrType();
      auto runeT = rewriter.getIntegerType(8);
      auto intT = this->getIntPtrType();
      const auto arrayT = mlir::LLVM::LLVMArrayType::get(runeT, strLen);

      // Get the pointer to the first character in the global string.
      Value globalPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, pointerT, name);
      Value addr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, pointerT, arrayT, globalPtr, ArrayRef<mlir::LLVM::GEPArg>{ 0, 0 });

      // Create the constant integer value representing this string's length.
      Value lenVal = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, this->getIntPtrType(), rewriter.getIntegerAttr(intT, strAttr.strref().size()));

      // Create the string struct
      mlir::Value structValue = rewriter.create<mlir::LLVM::UndefOp>(loc, resultType);
      structValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structValue, addr, 0);
      structValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structValue, lenVal, 1);

      // Replace the original operation with the string struct value.
      rewriter.replaceOp(op, structValue);
      return success();
    }
    return failure();
  }
};

struct CmpInterfaceOpLowering : ConvertOpToLLVMPattern<CmpInterfaceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    CmpInterfaceOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op->getLoc();
    const auto module = op->getParentOfType<mlir::ModuleOp>();

    // Determine which operand is the interface value.
    mlir::Value interfaceValue;
    mlir::Value otherValue;
    mlir::Type otherType;

    if (mlir::go::isa<mlir::go::InterfaceType>(op.getLhs().getType()))
    {
      interfaceValue = adaptor.getLhs();
      otherValue = adaptor.getRhs();
      otherType = op.getRhs().getType();
    }
    else
    {
      interfaceValue = adaptor.getRhs();
      otherValue = adaptor.getLhs();
      otherType = op.getLhs().getType();
    }

    // Create the runtime call to compare the interface value depending on the type of the "other"
    // value.
    mlir::Value result;
    if (mlir::go::isa<mlir::go::InterfaceType>(otherType))
    {
      // Emit the runtime call to do an interface to interface comparison.
      result = createRuntimeCall(
        rewriter,
        op.getLoc(),
        "interfaceCompare",
        this->getTypeConverter(),
        { interfaceValue, otherValue })[0];
    }
    else
    {
      mlir::Value addr;
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(op->getBlock());
        const mlir::Value oneValue =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
        addr = rewriter.create<mlir::LLVM::AllocaOp>(
          loc, this->getVoidPtrType(), otherValue.getType(), oneValue);
      }

      // Store a copy of the other value.
      // TODO: Need a slick way of getting the allocation associated with this value.
      rewriter.create<mlir::LLVM::StoreOp>(loc, otherValue, addr);

      // Get information about the other type.
      auto typeInfoGlobalOp = createTypeInfo(rewriter, module, loc, otherType);
      const Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);

      // Emit the runtime call to do an interface to arbitrary value comparison.
      result = createRuntimeCall(
        rewriter,
        loc,
        "interfaceCompareTo",
        this->getTypeConverter(),
        { interfaceValue, infoValue, addr })[0];
    }

    // Replace the operation.
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct CmpNilOpLowering : ConvertOpToLLVMPattern<CmpNilOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(CmpNilOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op->getLoc();

    // Replace with the respective runtime call to perform the nil comparison.
    const mlir::Value result =
      mlir::TypeSwitch<mlir::Type, mlir::Value>(mlir::go::baseType(op.getValue().getType()))
        .Case<mlir::go::SliceType>(
          [&](auto)
          {
            return createRuntimeCall(
              rewriter, loc, "sliceIsNil", this->getTypeConverter(), { adaptor.getValue() })[0];
          })
        .Case<mlir::go::MapType>(
          [&](auto)
          {
            return createRuntimeCall(
              rewriter, loc, "mapIsNil", this->getTypeConverter(), { adaptor.getValue() })[0];
          })
        .Default([&](auto) { return mlir::Value(); });

    if (!result)
    {
      return failure();
    }

    // Replace the operation.
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct CmpStringOpLowering : ConvertOpToLLVMPattern<CmpStringOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    CmpStringOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op->getLoc();

    // Replace with the runtime call to perform the string comparison.
    mlir::Value result = createRuntimeCall(
      rewriter,
      loc,
      "stringCompare",
      this->getTypeConverter(),
      { adaptor.getLhs(), adaptor.getRhs() })[0];

    // Replace the operation.
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct ZeroOpLowering : ConvertOpToLLVMPattern<ZeroOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(ZeroOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    auto resultType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct DeferOpLowering : ConvertOpToLLVMPattern<DeferOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(DeferOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    OpBuilder::InsertionGuard guard(rewriter);
    const Location loc = op.getLoc();

    mlir::Value fnValue =
      llvm::TypeSwitch<mlir::Type, mlir::Value>(mlir::go::baseType(op.getCallee().getType()))
        .Case([&](mlir::go::GoStructType) -> mlir::Value { return adaptor.getCallee(); });
    assert(fnValue && "func value is invalid");

    // Locate the head of the defer frame list for this function.
    //  mlir::Value deferStackPtrValue =
    //    locateOrCreateDeferStack(op, this->getTypeConverter(), rewriter);

    mlir::Value deferStackPtrValue;
    op->getParentOp()->walk(
      [&](mlir::Operation* op)
      {
        if (op->hasAttr("deferStack"))
        {
          deferStackPtrValue = rewriter.getRemappedValue(op->getResult(0));
        }
      });

    if (!deferStackPtrValue)
    {
      return op->emitOpError("no defer stack present in parent function");
    }

    // Create the runtime call to push the defer frame to the defer stack
    createRuntimeCall(
      rewriter, loc, "deferPush", this->getTypeConverter(), { deferStackPtrValue, fnValue });
    rewriter.eraseOp(op);

    return success();
  }
};

struct GetElementPointerOpLowering : ConvertOpToLLVMPattern<GetElementPointerOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    GetElementPointerOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto baseType = typeConverter->convertType(adaptor.getBaseType());
    const auto resultType = typeConverter->convertType(op.getType());
    SmallVector<mlir::LLVM::GEPArg> indices;
    for (auto index : adaptor.getConstIndices())
    {
      if (index & GetElementPointerOp::kValueFlag)
      {
        index = index & GetElementPointerOp::kValueIndexMask;
        indices.push_back(adaptor.getDynamicIndices()[index]);
      }
      else
      {
        indices.push_back(index);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, resultType, baseType, adaptor.getValue(), indices, false);
    return success();
  }
};

struct GlobalOpLowering : ConvertOpToLLVMPattern<GlobalOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(GlobalOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const Location loc = op.getLoc();
    auto linkage = adaptor.getAttributes().getAs<mlir::LLVM::LinkageAttr>("llvm.linkage");
    if (!linkage)
    {
      linkage = mlir::LLVM::LinkageAttr::get(this->getContext(), mlir::LLVM::Linkage::External);
    }

    auto elemT = this->getTypeConverter()->convertType(adaptor.getGlobalType());

    mlir::LLVM::DIGlobalVariableExpressionAttr diGlobalExprAttr;
    if (
      const auto fusedLoc =
        loc->findInstanceOf<mlir::FusedLocWith<mlir::LLVM::DIGlobalVariableExpressionAttr>>())
    {
      diGlobalExprAttr = fusedLoc.getMetadata();
    }

    // TODO: Any global that is NOT assigned a value in some function can be constant.
    mlir::SymbolRefAttr comdat;
    llvm::ArrayRef<mlir::NamedAttribute> attrs;
    auto global = rewriter.create<mlir::LLVM::GlobalOp>(
      loc,
      elemT,
      false,
      linkage.getLinkage(),
      adaptor.getSymName(),
      Attribute(),
      0,
      0,
      false,
      false,
      comdat,
      attrs,
      diGlobalExprAttr);

    // Copy the initializer regions.
    rewriter.inlineRegionBefore(op.getRegion(), global.getRegion(), global.getRegion().end());

    // Erase the old global op.
    rewriter.eraseOp(op);
    return success();
  }
};

struct GlobalCtorsOpLowering : ConvertOpToLLVMPattern<GlobalCtorsOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    GlobalCtorsOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    mlir::SmallVector<mlir::Attribute> data(
      adaptor.getCtors().size(), mlir::LLVM::ZeroAttr::get(this->getContext()));
    const auto dataAttr = mlir::ArrayAttr::get(this->getContext(), data);
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalCtorsOp>(
      op, adaptor.getCtors(), adaptor.getPriorities(), dataAttr);
    return success();
  }
};

struct GoOpLowering : ConvertOpToLLVMPattern<GoOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(GoOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    OpBuilder::InsertionGuard guard(rewriter);
    const mlir::Location loc = op.getLoc();

    mlir::Value fnValue =
      llvm::TypeSwitch<mlir::Type, mlir::Value>(mlir::go::baseType(op.getCallee().getType()))
        .Case([&](mlir::go::GoStructType) -> mlir::Value { return adaptor.getCallee(); });
    assert(fnValue && "func value is invalid");

    // Create the runtime call to push the defer frame to the defer stack
    createRuntimeCall(rewriter, loc, "addGoroutine", this->getTypeConverter(), { fnValue });
    rewriter.eraseOp(op);
    return success();
  }
};

struct IntToPtrOpLowering : ConvertOpToLLVMPattern<IntToPtrOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    IntToPtrOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto type = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, type, adaptor.getValue());
    return success();
  }
};

struct InterfaceCallOpLowering : ConvertOpToLLVMPattern<InterfaceCallOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    InterfaceCallOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    auto ptrType = this->getVoidPtrType();
    auto ifaceValue = adaptor.getIface();

    SmallVector<mlir::Type> argTypes = { ptrType };
    SmallVector<mlir::Type> resultTypes;

    // Compute method hash (method name, args types, result types)
    const auto signature = mlir::cast<FunctionType>(
      cast<InterfaceType>(op.getIface().getType()).getMethods().at(adaptor.getCallee().str()));
    auto methodHash = static_cast<uint32_t>(
      computeMethodHash(adaptor.getCallee(), signature.getInputs(), signature.getResults()));
    for (size_t i = 0; i < adaptor.getCalleeOperands().size(); ++i)
    {
      auto arg = adaptor.getCalleeOperands()[i];
      argTypes.push_back(this->getTypeConverter()->convertType(arg.getType()));
    }

    for (auto result : op->getResults())
    {
      resultTypes.push_back(this->getTypeConverter()->convertType(result.getType()));
    }

    // Create a constant int value for the hash
    auto constantIntOp =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), methodHash);
    auto constHashValue = constantIntOp.getResult();

    // Perform vtable lookup
    auto runtimeCallResults = createRuntimeCall(
      rewriter, loc, "interfaceLookUp", this->getTypeConverter(), { ifaceValue, constHashValue });

    const Value receiverValue = runtimeCallResults[0];
    const Value fnPtrValue = runtimeCallResults[1];

    // Collect the call arguments
    SmallVector<mlir::Value> callArgs = { receiverValue };
    for (auto arg : adaptor.getCalleeOperands())
    {
      callArgs.push_back(arg);
    }

    // Create the function type
    TypeConverter::SignatureConversion convResult(argTypes.size());
    auto fnT = rewriter.getFunctionType(argTypes, resultTypes);
    auto llvmFnT = mlir::cast<mlir::LLVM::LLVMFunctionType>(
      this->getTypeConverter()->convertFunctionSignature(fnT, false, false, convResult));

    // Perform indirect call
    SmallVector<Value> operands;
    operands.reserve(1 + callArgs.size());
    operands.push_back(fnPtrValue);
    append_range(operands, callArgs);

    auto newCallOp =
      rewriter.create<mlir::LLVM::CallOp>(loc, llvmFnT, FlatSymbolRefAttr(), operands);
    newCallOp.getProperties().operandSegmentSizes = { { static_cast<int32_t>(operands.size()),
                                                        0 } };
    newCallOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});

    SmallVector<Value, 4> results;
    if (resultTypes.size() < 2)
    {
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    }
    else
    {
      // Unpack the result values.
      results.reserve(resultTypes.size());
      for (size_t i = 0; i < resultTypes.size(); ++i)
      {
        results.push_back(
          rewriter.create<mlir::LLVM::ExtractValueOp>(loc, newCallOp->getResult(0), i));
      }
    }

    // Finally, replace the operation.
    rewriter.replaceOp(op, results);

    return success();
  }
};

struct LoadOpLowering : ConvertOpToLLVMPattern<LoadOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type = typeConverter->convertType(op.getType());
    const auto module = op->getParentOfType<ModuleOp>();
    const auto layout =
      llvm::DataLayout(mlir::dyn_cast<StringAttr>(module->getAttr("llvm.data_layout")));
    intptr_t alignment = static_cast<intptr_t>(layout.getPointerPrefAlignment().value());

    bool isVolatile = false;
    if (adaptor.getIsVolatile())
    {
      isVolatile = *adaptor.getIsVolatile();
    }

    mlir::LLVM::AtomicOrdering ordering = mlir::LLVM::AtomicOrdering::not_atomic;
    if (adaptor.getIsAtomic() && *adaptor.getIsAtomic())
    {
      ordering = mlir::LLVM::AtomicOrdering::acquire;
    }

    auto operand = adaptor.getOperand();
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
      op, type, operand, alignment, isVolatile, false, false, false, ordering);
    return success();
  }
};

struct MakeInterfaceOpLowering : ConvertOpToLLVMPattern<MakeInterfaceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    MakeInterfaceOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();

    // Get information about the dynamic type.
    auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), op.getDynamicType());
    const Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);

    // Lower to runtime call.
    const SmallVector<Value> args = { adaptor.getValue(), infoValue };
    const auto results =
      createRuntimeCall(rewriter, loc, "interfaceMake", this->getTypeConverter(), args);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MakeMapOpLowering : ConvertOpToLLVMPattern<MakeMapOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    MakeMapOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto dataLayout = mlir::DataLayout(module);
    const auto mapType = mlir::go::cast<MapType>(op.getType());
    const auto intType =
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed);

    // Get information about the key and element types.
    auto keyTypeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), mapType.getKeyType());
    auto elementTypeInfoGlobalOp =
      createTypeInfo(rewriter, module, op.getLoc(), mapType.getValueType());

    const Value keyTypeInfo = rewriter.create<mlir::LLVM::AddressOfOp>(loc, keyTypeInfoGlobalOp);
    const Value elementTypeInfo =
      rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
    const Value capacityValue = convert(
      rewriter,
      dataLayout,
      this->getTypeConverter(),
      adaptor.getCapacity(),
      op.getCapacity().getType(),
      intType,
      loc);

    // Lower to runtime call.
    const SmallVector<Value> args = { keyTypeInfo, elementTypeInfo, capacityValue };
    const auto results =
      createRuntimeCall(rewriter, loc, "interfaceMake", this->getTypeConverter(), args);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MakeSliceOpLowering : ConvertOpToLLVMPattern<MakeSliceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    MakeSliceOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto dataLayout = mlir::DataLayout(module);
    const auto sliceType = mlir::go::cast<SliceType>(op.getType());
    const auto intType =
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed);

    // Get information about the element type.
    auto elementTypeInfoGlobalOp =
      createTypeInfo(rewriter, module, op.getLoc(), sliceType.getElementType());

    const Value elementTypeInfo =
      rewriter.create<mlir::LLVM::AddressOfOp>(loc, elementTypeInfoGlobalOp);
    const Value lengthValue = convert(
      rewriter,
      dataLayout,
      this->getTypeConverter(),
      adaptor.getLength(),
      op.getLength().getType(),
      intType,
      loc);
    const Value capacityValue = convert(
      rewriter,
      dataLayout,
      this->getTypeConverter(),
      adaptor.getCapacity(),
      op.getCapacity().getType(),
      intType,
      loc);

    // Lower to runtime call.
    const SmallVector<Value> args = { elementTypeInfo, lengthValue, capacityValue };
    const auto results =
      createRuntimeCall(rewriter, loc, "sliceMake", this->getTypeConverter(), args);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MapLookupOpLowering : ConvertOpToLLVMPattern<MapLookupOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    MapLookupOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto mapType = mlir::go::cast<mlir::go::MapType>(op.getMap().getType());
    const auto keyType = this->getTypeConverter()->convertType(mapType.getKeyType());
    const auto elementType = this->getTypeConverter()->convertType(mapType.getValueType());

    // Store a copy of the key value on the stack.
    mlir::Value keyAddr;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
      keyAddr = rewriter.create<mlir::LLVM::AllocaOp>(
        loc,
        this->getVoidPtrType(),
        keyType,
        this->createIndexAttrConstant(rewriter, loc, rewriter.getI64Type(), 1));
    }
    rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getKey(), keyAddr);

    // Create runtime call to perform the map lookup.
    const auto results = createRuntimeCall(
      rewriter,
      loc,
      "mapLookup",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getMap(), keyAddr });

    mlir::Block* trueBlock;
    mlir::Block* falseBlock;
    mlir::Block* exitBlock = rewriter.splitBlock(op->getBlock(), std::next(op->getIterator()));
    exitBlock->addArgument(elementType, loc);

    // Build the true block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      trueBlock = rewriter.createBlock(op->getParentRegion(), exitBlock->getIterator());

      // Load the value.
      const mlir::Value value = rewriter.create<mlir::LLVM::LoadOp>(loc, elementType, results[0]);

      // Branch to the exit block.
      rewriter.create<mlir::LLVM::BrOp>(loc, mlir::SmallVector<mlir::Value>{ value }, exitBlock);
    }

    // Build the false block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      falseBlock = rewriter.createBlock(op->getParentRegion(), std::next(trueBlock->getIterator()));

      // Create the zero value.
      const mlir::Value value = rewriter.create<mlir::LLVM::ZeroOp>(loc, elementType);

      // Branch to the exit block.
      rewriter.create<mlir::LLVM::BrOp>(loc, mlir::SmallVector<mlir::Value>{ value }, exitBlock);
    }

    // Insert a conditional branch depending on if a value with the matching key was found.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<mlir::LLVM::CondBrOp>(loc, results[1], trueBlock, falseBlock);
    }

    mlir::SmallVector<mlir::Value> returnValues;
    returnValues.push_back(exitBlock->getArgument(0));
    if (op.getNumResults() == 2)
    {
      returnValues.push_back(results[1]);
    }

    // Replace the operation.
    rewriter.replaceOp(op, returnValues);

    return success();
  }
};

struct MapUpdateOpLowering : ConvertOpToLLVMPattern<MapUpdateOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    MapUpdateOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto mapType = mlir::go::cast<mlir::go::MapType>(op.getMap().getType());
    const auto keyType = this->getTypeConverter()->convertType(mapType.getKeyType());
    const auto elementType = this->getTypeConverter()->convertType(mapType.getValueType());

    // Store a copy of the key value on the stack.
    mlir::Value keyAddr;
    mlir::Value elementAddr;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
      keyAddr = rewriter.create<mlir::LLVM::AllocaOp>(
        loc,
        this->getVoidPtrType(),
        keyType,
        this->createIndexAttrConstant(rewriter, loc, rewriter.getI64Type(), 1));
      elementAddr = rewriter.create<mlir::LLVM::AllocaOp>(
        loc,
        this->getVoidPtrType(),
        elementType,
        this->createIndexAttrConstant(rewriter, loc, rewriter.getI64Type(), 1));
    }
    rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getKey(), keyAddr);
    rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getValue(), elementAddr);

    // Create runtime call to perform the map lookup.
    createRuntimeCall(
      rewriter,
      loc,
      "mapUpdate",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getMap(), keyAddr, elementAddr });

    // Remove the original operation.
    rewriter.eraseOp(op);

    return success();
  }
};

struct MapRangeOpLowering : public ConvertOpToLLVMPattern<MapRangeOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    MapRangeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto ptrType = this->getVoidPtrType();
    const auto block = op->getBlock();
    const auto mapType = mlir::go::dyn_cast<MapType>(op.getMap().getType());
    const auto keyType = this->getTypeConverter()->convertType(mapType.getKeyType());
    const auto elementType = this->getTypeConverter()->convertType(mapType.getValueType());

    // Create the runtime call to initialize an iterator for the range. Insert the call after the
    // map value is loaded.
    mlir::Value itAddr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(adaptor.getMap().getDefiningOp());

      auto iteratorValue = createRuntimeCall(
        rewriter, loc, "mapRangeInit", this->getTypeConverter(), { adaptor.getMap() })[0];

      {
        // Create a stack allocation to store the iterator in.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
        mlir::Value oneValue =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
        itAddr = rewriter.create<mlir::LLVM::AllocaOp>(
          loc, this->getVoidPtrType(), iteratorValue.getType(), oneValue);
      }

      // Store the iterator value.
      rewriter.create<mlir::LLVM::StoreOp>(loc, iteratorValue, itAddr);
    }

    // Replace the range op with the respective runtime call.
    auto results =
      createRuntimeCall(rewriter, loc, "mapRange", this->getTypeConverter(), { itAddr });
    rewriter.eraseOp(op);

    mlir::Value keyAddrValue = results[0];
    mlir::Value elementAddrValue = results[1];
    mlir::Value okValue = results[2];

    // Build the load block.
    mlir::Block* loadBlock;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      loadBlock = rewriter.createBlock(
        block->getParent(),
        std::next(block->getIterator()),
        mlir::SmallVector<mlir::Type>{ ptrType, ptrType },
        mlir::SmallVector<mlir::Location>{ loc, loc });
      Value keyValue = rewriter.create<mlir::LLVM::LoadOp>(loc, keyType, loadBlock->getArgument(0));
      Value elementValue =
        rewriter.create<mlir::LLVM::LoadOp>(loc, elementType, loadBlock->getArgument(1));
      rewriter.create<mlir::LLVM::BrOp>(
        loc, SmallVector<mlir::Value>{ keyValue, elementValue }, op.getBodyBlock());
    }

    // Conditionally branch to the load block if the range iteration was successful. Otherwise,
    // branch to the exit block.
    rewriter.create<mlir::LLVM::CondBrOp>(
      loc,
      okValue,
      loadBlock,
      SmallVector<mlir::Value>{ keyAddrValue, elementAddrValue },
      op.getExitBlock(),
      SmallVector<mlir::Value>{});

    return success();
  }
};

struct PanicOpLowering : ConvertOpToLLVMPattern<PanicOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(PanicOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();

    // Create the runtime call to schedule this function call
    createRuntimeCall(rewriter, loc, "_panic", this->getTypeConverter(), { adaptor.getValue() });

    // The end of the function should be unreachable
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);

    return success();
  }
};

struct PointerToFunctionOpLowering : ConvertOpToLLVMPattern<PointerToFunctionOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    PointerToFunctionOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto type = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, type, adaptor.getValue());
    return success();
  }
};

struct PtrToIntOpLowering : ConvertOpToLLVMPattern<PtrToIntOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    PtrToIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto type = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, type, adaptor.getValue());
    return success();
  }
};

struct RecoverOpLowering : ConvertOpToLLVMPattern<RecoverOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    RecoverOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto runtimeCallResults =
      createRuntimeCall(rewriter, loc, "_recover", this->getTypeConverter(), {});
    rewriter.replaceOp(op, runtimeCallResults);
    return success();
  }
};

struct RecvOpLowering : ConvertOpToLLVMPattern<RecvOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(RecvOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();
    auto type = typeConverter->convertType(op.getType());
    // Allocate stack to receive the value into
    auto arrSizeConstOp = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerType::get(rewriter.getContext(), 64), 1);
    auto allocaOp = rewriter.create<mlir::LLVM::AllocaOp>(loc, type, arrSizeConstOp.getResult());

    // Create the runtime call to receive a value from the channel
    auto blockConstOp = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerType::get(rewriter.getContext(), 64), adaptor.getCommaOk() ? 1 : 0);
    createRuntimeCall(
      rewriter,
      loc,
      "_channelReceive",
      this->getTypeConverter(),
      {
        op.getOperand(),
        allocaOp.getResult(),
        blockConstOp.getResult(),
      });
    // Load the value
    auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, type, allocaOp.getResult());
    rewriter.replaceOp(op, loadOp->getResults());
    return success();
  }
};

struct RunDefersLowering : ConvertOpToLLVMPattern<RunDefersOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    RunDefersOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();

    // Locate the head of the defer frame list for this function.
    mlir::Value deferStackPtrValue =
      locateOrCreateDeferStack(op, this->getTypeConverter(), rewriter);

    // Create the runtime call to run the defer stack.
    createRuntimeCall(rewriter, loc, "deferRun", this->getTypeConverter(), { deferStackPtrValue });

    // Erase the original operation.
    rewriter.eraseOp(op);

    return success();
  }
};

struct StringAddrOpLowering : ConvertOpToLLVMPattern<StringAddrOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    StringAddrOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const DataLayout dataLayout(module);
    const auto originalIndexType = op.getIndex().getType();
    const auto expectedIndexType =
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed);

    mlir::Value indexValue = convert(
      rewriter,
      dataLayout,
      this->getTypeConverter(),
      adaptor.getIndex(),
      originalIndexType,
      expectedIndexType,
      loc);

    // Replace with runtime call.
    mlir::Value result = createRuntimeCall(
      rewriter,
      loc,
      "stringIndexAddr",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getValue(), indexValue })[0];
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct StringRangeOpLowering : public ConvertOpToLLVMPattern<StringRangeOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
    StringRangeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();

    // Create the runtime call to initialize an iterator for the range. Insert the call after the
    // string value is loaded.
    mlir::Value itAddr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(adaptor.getValue().getDefiningOp());

      auto iteratorValue = createRuntimeCall(
        rewriter, loc, "stringRangeInit", this->getTypeConverter(), { adaptor.getValue() })[0];

      {
        // Create a stack allocation to store the iterator in.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
        mlir::Value oneValue =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
        itAddr = rewriter.create<mlir::LLVM::AllocaOp>(
          loc, this->getVoidPtrType(), iteratorValue.getType(), oneValue);
      }

      // Store the iterator value.
      rewriter.create<mlir::LLVM::StoreOp>(loc, iteratorValue, itAddr);
    }

    // Replace the range op with the respective runtime call.
    auto results =
      createRuntimeCall(rewriter, loc, "stringRange", this->getTypeConverter(), { itAddr });
    rewriter.eraseOp(op);

    // Conditionally branch to the body block if the range iteration was successful. Otherwise,
    // branch to the exit block.
    rewriter.create<mlir::LLVM::CondBrOp>(
      loc,
      results[2],
      op.getBodyDest(),
      SmallVector<mlir::Value>{ results[0], results[1] },
      op.getExitBlock(),
      SmallVector<mlir::Value>{});

    return success();
  }
};

struct StringToSliceOpLowering : ConvertOpToLLVMPattern<StringToSliceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    StringToSliceOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();

    // Replace with runtime call.
    mlir::Value result = createRuntimeCall(
      rewriter,
      loc,
      "stringToSlice",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getValue() })[0];
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SliceOpLowering : ConvertOpToLLVMPattern<SliceOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SliceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto dataLayout = mlir::DataLayout(module);
    const auto expectedIndexType =
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed);

    mlir::Value lowValue;
    mlir::Value highValue;
    mlir::Value maxValue;

    // Perform index type conversions.
    if (op.getLow())
    {
      lowValue = convert(
        rewriter,
        dataLayout,
        this->getTypeConverter(),
        adaptor.getLow(),
        op.getLow().getType(),
        expectedIndexType,
        loc);
    }

    if (op.getHigh())
    {
      highValue = convert(
        rewriter,
        dataLayout,
        this->getTypeConverter(),
        adaptor.getHigh(),
        op.getHigh().getType(),
        expectedIndexType,
        loc);
    }

    if (op.getMax())
    {
      maxValue = convert(
        rewriter,
        dataLayout,
        this->getTypeConverter(),
        adaptor.getMax(),
        op.getMax().getType(),
        expectedIndexType,
        loc);
    }

    // Replace the operation with the respective runtime call.
    mlir::Value result =
      mlir::TypeSwitch<mlir::Type, mlir::Value>(op.getInput().getType())
        .Case(
          [&](mlir::go::SliceType T)
          {
            // Get the element type information.
            const auto elementType = T.getElementType();
            auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), elementType);
            const Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);

            // Gather runtime call arguments.
            SmallVector<Value, 5> args = {
              adaptor.getInput(),
              infoValue,
              op.getLow() ? lowValue
                          : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1),
              op.getHigh()
                ? highValue
                : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1),
              op.getMax() ? maxValue
                          : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1)
            };
            return createRuntimeCall(
              rewriter, loc, "sliceReslice", this->getTypeConverter(), args)[0];
          })
        .Case(
          [&](mlir::go::StringType)
          {
            // Gather runtime call arguments.
            SmallVector<Value, 5> args = {
              adaptor.getInput(),
              op.getLow() ? lowValue
                          : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1),
              op.getHigh() ? highValue
                           : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1)
            };
            return createRuntimeCall(
              rewriter, loc, "stringSlice", this->getTypeConverter(), args)[0];
          })
        .Case(
          [&](mlir::go::PointerType T)
          {
            const auto arrayT = go::cast<ArrayType>(*T.getElementType());
            const auto elementT = arrayT.getElementType();
            const auto length = arrayT.getLength();
            const auto stride = dataLayout.getTypeSize(elementT);
            const auto uintptrType = this->getTypeConverter()->convertType(
              mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Uintptr));

            // Gather runtime call arguments.
            SmallVector<Value, 5> args = {
              adaptor.getInput(),
              op.getLow() ? lowValue
                          : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1),
              op.getHigh()
                ? highValue
                : rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), -1),
              rewriter.create<mlir::LLVM::ConstantOp>(
                loc, this->getTypeConverter()->convertType(expectedIndexType), length),
              rewriter.create<mlir::LLVM::ConstantOp>(
                loc, this->getTypeConverter()->convertType(uintptrType), stride)
            };
            return createRuntimeCall(rewriter, loc, "sliceAddr", this->getTypeConverter(), args)[0];
          });

    // Replace the operation.
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SliceAddrOpLowering : ConvertOpToLLVMPattern<SliceAddrOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    SliceAddrOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto dataLayout = mlir::DataLayout(module);
    const auto sliceType = mlir::go::cast<SliceType>(op.getSlice().getType());
    const auto elementType = sliceType.getElementType();
    const auto originalIndexType = op.getIndex().getType();
    const auto expectedIndexType =
      mlir::go::IntegerType::get(this->getContext(), mlir::go::IntegerType::Signed);

    // Get information about the element type.
    auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), elementType);
    const Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);

    mlir::Value indexValue = convert(
      rewriter,
      dataLayout,
      this->getTypeConverter(),
      adaptor.getIndex(),
      originalIndexType,
      expectedIndexType,
      loc);

    // Replace with runtime call.
    const mlir::Value result = createRuntimeCall(
      rewriter,
      loc,
      "sliceIndexAddr",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getSlice(), indexValue, infoValue })[0];
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct SliceToStringOpLowering : ConvertOpToLLVMPattern<SliceToStringOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    SliceToStringOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();

    // Replace with runtime call.
    mlir::Value result = createRuntimeCall(
      rewriter,
      loc,
      "sliceToString",
      this->getTypeConverter(),
      mlir::SmallVector<mlir::Value>{ adaptor.getValue() })[0];
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct StoreOpLowering : ConvertOpToLLVMPattern<StoreOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    auto addr = adaptor.getAddr();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto layout =
      llvm::DataLayout(mlir::dyn_cast<mlir::StringAttr>(module->getAttr("llvm.data_layout")));
    intptr_t alignment = static_cast<intptr_t>(layout.getPointerPrefAlignment().value());

    auto value = adaptor.getValue();

    bool isVolatile = false;
    if (adaptor.getIsVolatile())
    {
      isVolatile = *adaptor.getIsVolatile();
    }

    mlir::LLVM::AtomicOrdering ordering = mlir::LLVM::AtomicOrdering::not_atomic;
    if (adaptor.getIsAtomic() && *adaptor.getIsAtomic())
    {
      ordering = mlir::LLVM::AtomicOrdering::release;
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
      op, value, addr, alignment, isVolatile, false, false, ordering);
    return success();
  }
};

struct TypeAssertOpLowering : ConvertOpToLLVMPattern<TypeAssertOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    TypeAssertOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    const auto module = op->getParentOfType<ModuleOp>();
    const auto valueType = this->getTypeConverter()->convertType(op.getType(0));

    SmallVector<mlir::Value> args;
    args.push_back(adaptor.getValue());

    // Create the type information for the asserted type.
    auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), op.getType(0));
    const mlir::Value infoPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, typeInfoGlobalOp);
    args.push_back(infoPtr);

    // Assert the hasOk flag if the second result is present.
    const mlir::Value hasOk = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerType::get(rewriter.getContext(), 1), op.getNumResults() == 2 ? 1 : 0);
    args.push_back(hasOk);

    // Create runtime call to perform the type assertion.
    auto results =
      createRuntimeCall(rewriter, loc, "interfaceAssert", this->getTypeConverter(), args);

    // Create blocks for handling the result.
    mlir::Block* trueDest;
    mlir::Block* falseDest;

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      trueDest =
        rewriter.createBlock(op->getParentRegion(), std::next(op->getBlock()->getIterator()));
    }

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      falseDest = rewriter.createBlock(op->getParentRegion(), std::next(trueDest->getIterator()));
    }

    // Conditionally branch to either the true block or the false block.
    rewriter.create<mlir::LLVM::CondBrOp>(loc, results[1], trueDest, falseDest);

    // Split the type assert operation into the exit block.
    mlir::Block* exitDest = rewriter.splitBlock(op->getBlock(), op->getIterator());
    exitDest->addArgument(valueType, loc);

    // Build the true block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(trueDest);

      mlir::Value value;
      if (mlir::go::isa<mlir::go::InterfaceType>(op.getType(0)))
      {
        // Return a new interface value.
        value = createRuntimeCall(
          rewriter,
          loc,
          "interfaceMake",
          this->getTypeConverter(),
          mlir::SmallVector<mlir::Value>{ results[0], infoPtr })[0];
      }
      else
      {
        // Load the value.
        value = rewriter.create<mlir::LLVM::LoadOp>(loc, valueType, results[0]);
      }

      // Branch to the exit block passing the value.
      rewriter.create<mlir::LLVM::BrOp>(loc, mlir::SmallVector<mlir::Value>{ value }, exitDest);
    }

    // Build the false block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(falseDest);

      const mlir::Value value = rewriter.create<mlir::LLVM::ZeroOp>(loc, valueType);

      // Branch to the exit block passing the value.
      rewriter.create<mlir::LLVM::BrOp>(loc, mlir::SmallVector<mlir::Value>{ value }, exitDest);
    }

    SmallVector<mlir::Value> returnValues = { exitDest->getArgument(0) };
    if (op.getNumResults() == 2)
    {
      returnValues.push_back(results[1]);
    }

    // The block argument replaces the operation.
    rewriter.replaceOp(op, returnValues);

    return success();
  }
};

struct UnrealizedConversionCastOpLowering : ConvertOpToLLVMPattern<mlir::UnrealizedConversionCastOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    UnrealizedConversionCastOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    /*
    const auto inputType = op.getInputs()[0].getType();
    for (auto user : op->getUsers())
    {
      if (mlir::isa<mlir::UnrealizedConversionCastOp>(user))
      {
        const auto userResultType = user->getResult(0).getType();
        inputType.dump();
        userResultType.dump();
        if (userResultType == inputType)
        {
          // Replace the user.
          rewriter.replaceOp(user, op.getInputs()[0]);
        }
      }
    }
    // Erase this operation.
    rewriter.eraseOp(op);
    */

    rewriter.replaceOp(op, op.getInputs());

    return success();
  }
};

struct ExtractOpLowering : ConvertOpToLLVMPattern<ExtractOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ExtractOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
      op, resultType, adaptor.getAggregate(), adaptor.getIndex());
    return success();
  }
};

struct InlineAsmOpLowering final : ConvertOpToLLVMPattern<InlineAsmOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  auto matchAndRewrite(InlineAsmOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const
    -> LogicalResult override
  {
    mlir::Type resultType;
    mlir::SmallVector<std::string> clobbers;
    mlir::SmallVector<mlir::Attribute> operandAttrs;
    mlir::SmallVector<mlir::Type> outputTypes;
    mlir::SmallVector<mlir::Value> outputPtrValues;
    mlir::SmallVector<mlir::Value> inputValues;
    mlir::DenseMap<size_t, size_t> indexMap;

    const auto loc = op.getLoc();

    mlir::SmallVector<mlir::go::AsmConstraintAttr> constraints;
    auto outputsIt = constraints.begin();

    // Sort the constraints.
    for (const auto& c : op.getConstraints())
    {
      const auto constraintAttr = mlir::cast<mlir::go::AsmConstraintAttr>(c);
      if (constraintAttr.getDirection().getOutput())
      {
        // Outputs go first.
        outputsIt = constraints.insert(outputsIt, constraintAttr) + 1;
      }
      else if (constraintAttr.getDirection().getInput())
      {
        // Input go after outputs.
        constraints.push_back(constraintAttr);
      }
    }

    mlir::SmallVector<std::string> constraintsCodes;
    size_t constraintIndexOffset = 0;
    for (size_t i = 0; i < constraints.size(); ++i)
    {
      const auto constraintAttr = mlir::cast<mlir::go::AsmConstraintAttr>(constraints[i]);
      std::string code = constraintAttr.getRegisterClass().str();

      if (constraintAttr.getDirection().getOutput())
      {
        std::string modifier = "=";
        if (constraintAttr.getReserve())
        {
          // Add the early clobber modifier.
          modifier += "&";
        }

        // Prepend the modifier to the constraint code.
        code = modifier + code;

        if (constraintAttr.getDirection().getInput())
        {
          code = code + "," + std::to_string(i + constraintIndexOffset);
          constraintIndexOffset++;

          if (constraintAttr.getOperandIndex())
          {
            const auto index = constraintAttr.getOperandIndex().getInt();
            mlir::Value operand = adaptor.getOperandValues()[index];
            inputValues.push_back(operand);
          }
        }

        if (constraintAttr.getOperandIndex())
        {
          const auto index = constraintAttr.getOperandIndex().getInt();
          mlir::Value operand = op.getOperandValues()[index];
          outputTypes.push_back(operand.getType());
          outputPtrValues.push_back(adaptor.getOperandValues()[index]);
        }

        constraintsCodes.push_back(code);
      }
      else if (constraintAttr.getDirection().getInput())
      {
        if (constraintAttr.getOperandIndex())
        {
          const auto index = constraintAttr.getOperandIndex().getInt();
          mlir::Value operand = adaptor.getOperandValues()[index];
          inputValues.push_back(operand);
        }
        constraintsCodes.push_back(code);
      }

      indexMap[i] = i + constraintIndexOffset;
    }

    // Add clobbers last.
    for (const auto& attr : op.getRegisterClobbers())
    {
      const auto strAttr = mlir::cast<mlir::StringAttr>(attr);
      // Format the register string.
      std::string code = "~{" + strAttr.str() + "}";
      constraintsCodes.push_back(code);
    }

    // Format the constraint code string.
    const std::string constraintsStr =
      llvm::join(constraintsCodes.begin(), constraintsCodes.end(), ",");

    if (outputTypes.size() > 1)
    {
      mlir::SmallVector<mlir::Type> resultStructTypes;
      for (size_t i = 0; i < outputTypes.size(); ++i)
      {
        const auto outputType = mlir::go::dyn_cast<mlir::go::PointerType>(outputTypes[i]);
        resultStructTypes.push_back(typeConverter->convertType(*outputType.getElementType()));
      }
      resultType = mlir::LLVM::LLVMStructType::getLiteral(this->getContext(), resultStructTypes);
    }
    else if (outputTypes.size() == 1)
    {
      const auto outputType = mlir::go::dyn_cast<mlir::go::PointerType>(outputTypes[0]);
      if (outputType.getElementType().has_value())
      {
        resultType = typeConverter->convertType(*outputType.getElementType());
      }
      else
      {
        resultType = this->getVoidPtrType();
      }
    }

    // Substitute the aliases in the assembly string with their indices.
    std::string asmStr = op.getAsmString().str();
    for (size_t i = 0; i < constraints.size(); ++i)
    {
      const auto constraintAttr = mlir::cast<mlir::go::AsmConstraintAttr>(constraints[i]);
      if (constraintAttr.getAlias())
      {
        const std::string aliasStr = "{{" + constraintAttr.getAlias().str() + "}}";
        const std::string indexStr = "$" + std::to_string(indexMap[i]);
        stringReplaceAll(asmStr, aliasStr, indexStr);
      }
    }

    const auto asmStrAttr = mlir::StringAttr::get(this->getContext(), asmStr);

    auto inlineAsmOp = rewriter.create<mlir::LLVM::InlineAsmOp>(
      loc,
      resultType,
      inputValues,
      asmStrAttr,
      constraintsStr,
      true,
      true,
      mlir::LLVM::AsmDialectAttr(),
      mlir::ArrayAttr());

    if (outputTypes.size() > 1)
    {
      for (int64_t i = 0; i < outputPtrValues.size(); ++i)
      {
        mlir::Value resultValue = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, inlineAsmOp.getResult(0), mlir::SmallVector<int64_t>{ i });
        rewriter.create<mlir::LLVM::StoreOp>(loc, resultValue, outputPtrValues[i]);
      }
    }
    else if (outputTypes.size() == 1)
    {
      rewriter.create<mlir::LLVM::StoreOp>(loc, inlineAsmOp.getResult(0), outputPtrValues[0]);
    }

    // Remove the original operation.
    rewriter.eraseOp(op);

    return success();
  }
};

struct InsertOpLowering : ConvertOpToLLVMPattern<InsertOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(InsertOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
      op, resultType, adaptor.getAggregate(), adaptor.getValue(), adaptor.getIndex());
    return success();
  }
};

struct YieldOpLowering : ConvertOpToLLVMPattern<YieldOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(YieldOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, adaptor.getInitializerValue());
    return success();
  }
};
} // namespace transforms::LLVM

void populateGoToLLVMConversionPatterns(
  mlir::LLVMTypeConverter& converter,
  RewritePatternSet& patterns)
{
  // clang-format off
        patterns.add<
            transforms::LLVM::AddressOfOpLowering,
            transforms::LLVM::AddStrOpLowering,
            transforms::LLVM::AllocaOpLowering,
            transforms::LLVM::AtomicAddIOpLowering,
            transforms::LLVM::AtomicCompareAndSwapIOpLowering,
            transforms::LLVM::AtomicSwapIOpLowering,
            transforms::LLVM::BitcastOpLowering,
            transforms::LLVM::BuiltInCallOpLowering,
            transforms::LLVM::CallIndirectOpLowering,
            transforms::LLVM::ChangeInterfaceOpLowering,
            transforms::LLVM::ChanRangeOpLowering,
            transforms::LLVM::ChanSelectOpLowering,
            transforms::LLVM::ChanSendOpLowering,
            transforms::LLVM::ChanRecvOpLowering,
            transforms::LLVM::ConstantOpLowering,
            transforms::LLVM::CmpInterfaceOpLowering,
            transforms::LLVM::CmpNilOpLowering,
            transforms::LLVM::CmpStringOpLowering,
            transforms::LLVM::DeferOpLowering,
            transforms::LLVM::ExtractOpLowering,
            transforms::LLVM::GetElementPointerOpLowering,
            transforms::LLVM::GlobalOpLowering,
            transforms::LLVM::GlobalCtorsOpLowering,
            transforms::LLVM::GoOpLowering,
            transforms::LLVM::InlineAsmOpLowering,
            transforms::LLVM::InsertOpLowering,
            transforms::LLVM::InterfaceCallOpLowering,
            transforms::LLVM::IntToPtrOpLowering,
            transforms::LLVM::LoadOpLowering,
            transforms::LLVM::MakeInterfaceOpLowering,
            transforms::LLVM::MakeMapOpLowering,
            transforms::LLVM::MakeSliceOpLowering,
            transforms::LLVM::MapLookupOpLowering,
            transforms::LLVM::MapUpdateOpLowering,
            transforms::LLVM::MapRangeOpLowering,
            transforms::LLVM::PanicOpLowering,
            transforms::LLVM::PointerToFunctionOpLowering,
            transforms::LLVM::PtrToIntOpLowering,
            transforms::LLVM::RecoverOpLowering,
            transforms::LLVM::RecvOpLowering,
            transforms::LLVM::RunDefersLowering,
            transforms::LLVM::SliceOpLowering,
            transforms::LLVM::SliceAddrOpLowering,
            transforms::LLVM::SliceToStringOpLowering,
            transforms::LLVM::StringAddrOpLowering,
            transforms::LLVM::StringRangeOpLowering,
            transforms::LLVM::StringToSliceOpLowering,
            transforms::LLVM::StoreOpLowering,
            transforms::LLVM::TypeAssertOpLowering,
            transforms::LLVM::TypeAssertOpLowering,
            transforms::LLVM::UnrealizedConversionCastOpLowering,
            transforms::LLVM::YieldOpLowering,
            transforms::LLVM::ZeroOpLowering
        >(converter);
        // clang-format off
    }
} // namespace mlir::go
