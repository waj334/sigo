
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/DLTI/Traits.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeInfo.h"
#include "Go/Util.h"

namespace mlir::go
{

mlir::Type lookUpRuntimeType(mlir::ModuleOp module, mlir::StringRef name)
{
  auto moduleOp = module.getOperation();
  assert(
    moduleOp->hasAttr("go.runtimeTypes") &&
    "module MUST have the go.runtimeTypes dictionary attribute");
  auto typeMap = ::mlir::dyn_cast<mlir::DictionaryAttr>(moduleOp->getAttr("go.runtimeTypes"));
  const ::mlir::TypeAttr result = typeMap.template getAs<::mlir::TypeAttr>(name);
  if (!result)
  {
    return ::mlir::Type();
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

static Value createParameterPack(
  PatternRewriter& rewriter,
  const Location location,
  const ArrayRef<Value>& params,
  intptr_t& size,
  const DataLayout& layout,
  const mlir::LLVMTypeConverter* converter)
{
  auto wordType = mlir::IntegerType::get(rewriter.getContext(), converter->getPointerBitwidth());

  /*
  {
      intptr_t numArgs
      intptr_t SIZEOF(ARG0)
      [ARG0]
      ...
      intptr_t SIZEOF(ARGN)
      [ARGN]
  }
  */

  // Create the context struct type
  SmallVector<Type> elementTypes = { wordType };

  // Append parameter types
  for (auto param : params)
  {
    elementTypes.push_back(wordType);
    elementTypes.push_back(param.getType());
  }
  auto packType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elementTypes);

  // Create an undef of the parameter pack struct type
  Value packContainerValue = rewriter.create<LLVM::UndefOp>(location, packType);

  // Set the argument count in the parameter pack
  auto constantIntOp = rewriter.create<LLVM::ConstantOp>(location, wordType, params.size());
  packContainerValue = rewriter.create<LLVM::InsertValueOp>(
    location, packContainerValue, constantIntOp.getResult(), 0);

  // Populate the arguments parameter pack struct
  int64_t index = 1;
  for (auto param : params)
  {
    // Insert the size value
    const auto paramSize = layout.getTypeSize(param.getType());
    constantIntOp = rewriter.create<LLVM::ConstantOp>(location, wordType, paramSize);
    packContainerValue = rewriter.create<LLVM::InsertValueOp>(
      location, packContainerValue, constantIntOp.getResult(), index++);

    // Insert the argument value
    packContainerValue =
      rewriter.create<LLVM::InsertValueOp>(location, packContainerValue, param, index++);
  }

  // Get the allocated size of the parameter pack struct
  size = (intptr_t)layout.getTypeSize(packType);

  return packContainerValue;
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
    Block& entryBlock = *parentFunc.getBody().begin();
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
      if (!op->getBlock()->isEntryBlock())
      {
        // Create the alloca operation for the stack allocation in the entry block of its respective
        // function.
        rewriter.setInsertionPointToStart(&entryBlock);
      }

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
    }

    // Create debug information if set on the operation.
    if (
      const auto fusedLoc =
        loc->findInstanceOf<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>())
    {
      rewriter.create<mlir::LLVM::DbgDeclareOp>(
        allocValue.getDefiningOp()->getLoc(),
        allocValue,
        fusedLoc.getMetadata(),
        mlir::LLVM::DIExpressionAttr());
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
    else if (callee == "max")
    {
      Value incomingValue = operands[0];
      Type valueType = incomingValue.getType();

      // Add the block parameter that will be used to receive the largest value.
      Block* successor = rewriter.getBlock();
      Value result = successor->addArgument(valueType, incomingValue.getLoc());
      rewriter.replaceOp(op, { result });

      // TODO: Implement fast path for scenario where all values are constants.

      if (operands.size() == 1)
      {
        rewriter.replaceOp(op, { incomingValue });
      }
      else
      {
        SmallVector<mlir::Block*> blocks;

        // Create the initial predecessor block.
        rewriter.createBlock(rewriter.getBlock());

        // Create blocks.
        for (size_t i = 1; i < operands.size(); ++i)
        {
          const Value nextValue = operands[i];

          // Compare the incoming value against the next value using the respective comparison
          // operation.
          Value cond =
            TypeSwitch<Type, Value>(valueType)
              .Case(
                [&](IntegerType) -> Value
                {
                  const auto predicate = isUnsigned(op->getOperandTypes()[0])
                    ? mlir::LLVM::ICmpPredicate::ugt
                    : mlir::LLVM::ICmpPredicate::sgt;
                  return rewriter.create<mlir::LLVM::ICmpOp>(
                    loc, boolType, predicate, incomingValue, nextValue);
                })
              .Case(
                [&](FloatType) -> Value
                {
                  return rewriter.create<mlir::LLVM::FCmpOp>(
                    loc, boolType, mlir::LLVM::FCmpPredicate::ogt, incomingValue, nextValue);
                });

          mlir::Block* next;

          if (i < operands.size() - 1)
          {
            // Create the next block to jump to.
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            next = rewriter.createBlock(rewriter.getBlock(), valueType, { operands[i].getLoc() });
          }
          else
          {
            next = successor;
          }

          // Pass the larger value to the next block to perform the next comparison with.
          rewriter.create<mlir::LLVM::CondBrOp>(
            loc,
            cond,
            next,
            SmallVector<Value>{ incomingValue },
            next,
            SmallVector<Value>{ nextValue });

          // Continue insertion in the next block.
          incomingValue = next->getArgument(0);
          rewriter.setInsertionPointToStart(next);
        }
      }
    }
    else if (callee == "min")
    {
      Value incomingValue = operands[0];
      Type valueType = incomingValue.getType();

      // Add the block parameter that will be used to receive the smallest value.
      Block* successor = rewriter.getBlock();
      Value result = successor->addArgument(valueType, incomingValue.getLoc());
      rewriter.replaceOp(op, { result });

      // TODO: Implement fast path for scenario where all values are constants.

      if (operands.size() == 1)
      {
        rewriter.replaceOp(op, { incomingValue });
      }
      else
      {
        SmallVector<mlir::Block*> blocks;

        // Create the initial predecessor block.
        rewriter.createBlock(rewriter.getBlock());

        // Create blocks.
        for (size_t i = 1; i < operands.size(); ++i)
        {
          const Value nextValue = operands[i];

          // Compare the incoming value against the next value using the respective comparison
          // operation.
          Value cond =
            TypeSwitch<Type, Value>(valueType)
              .Case(
                [&](IntegerType) -> Value
                {
                  const auto predicate = isUnsigned(op->getOperandTypes()[0])
                    ? mlir::LLVM::ICmpPredicate::ult
                    : mlir::LLVM::ICmpPredicate::slt;
                  return rewriter.create<mlir::LLVM::ICmpOp>(
                    loc, boolType, predicate, incomingValue, nextValue);
                })
              .Case(
                [&](FloatType) -> Value
                {
                  return rewriter.create<mlir::LLVM::FCmpOp>(
                    loc, boolType, mlir::LLVM::FCmpPredicate::olt, incomingValue, nextValue);
                });

          mlir::Block* next;

          if (i < operands.size() - 1)
          {
            // Create the next block to jump to.
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            next = rewriter.createBlock(rewriter.getBlock(), valueType, { operands[i].getLoc() });
          }
          else
          {
            next = successor;
          }

          // Pass the smaller value to the next block to perform the next comparison with.
          rewriter.create<mlir::LLVM::CondBrOp>(
            loc,
            cond,
            next,
            SmallVector<Value>{ incomingValue },
            next,
            SmallVector<Value>{ nextValue });

          // Continue insertion in the next block.
          incomingValue = next->getArgument(0);
          rewriter.setInsertionPointToStart(next);
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
    for (auto i = 0; i < op.getNumResults(); i++)
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
    Location loc = op.getLoc();

    const auto module = op->getParentOfType<ModuleOp>();
    auto dataLayout = mlir::DataLayout(module);
    auto wordType =
      mlir::IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());

    // Create the parameter pack holding the arguments
    intptr_t packSize;
    auto pack = createParameterPack(
      rewriter,
      loc,
      llvm::SmallVector<mlir::Value>(adaptor.getCalleeOperands()),
      packSize,
      dataLayout,
      getTypeConverter());

    // Allocate memory on the heap to store the parameter pack into
    auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(loc, wordType, packSize);
    auto runtimeCallResults = createRuntimeCall(
      rewriter, loc, "alloc", this->getTypeConverter(), { constantIntOp.getResult() });

    // Store the parameter pack into the allocated memory
    rewriter.create<mlir::LLVM::StoreOp>(loc, pack, runtimeCallResults[0]);

    // Create the runtime call to push the defer frame to the defer stack
    createRuntimeCall(
      rewriter, loc, "deferPush", {}, { adaptor.getCallee(), runtimeCallResults[0] });

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
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalCtorsOp>(
      op, adaptor.getCtors(), adaptor.getPriorities());
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
    auto methodHash = uint32_t(computeMethodHash(adaptor.getCallee(), signature, true));
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
      op, type, operand, alignment, isVolatile, false, false, ordering);
    return success();
  }
};

struct MakeOpLowering : ConvertOpToLLVMPattern<MakeOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MakeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();
    auto baseType = underlyingType(op.getType());
    SmallVector<mlir::Value> runtimeCallResults;

    if (mlir::isa<ChanType>(baseType))
    {
      SmallVector<Value> args;
      if (op->getNumOperands() > 0)
      {
        args = adaptor.getOperands();
      }
      else
      {
        auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, mlir::IntegerType::get(rewriter.getContext(), 32), 0);
        args = { constantIntOp.getResult() };
      }
      runtimeCallResults =
        createRuntimeCall(rewriter, loc, "chanMake", this->getTypeConverter(), args);
    }
    else if (mlir::isa<SliceType>(baseType))
    {
      SmallVector<Value> args = adaptor.getOperands();
      runtimeCallResults =
        createRuntimeCall(rewriter, loc, "sliceMake", this->getTypeConverter(), args);
    }
    else if (mlir::isa<MapType>(baseType))
    {
      SmallVector<Value> args;
      if (op->getNumOperands() > 0)
      {
        args = adaptor.getOperands();
      }
      else
      {
        auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, mlir::IntegerType::get(rewriter.getContext(), 32), 0);
        args = { constantIntOp.getResult() };
      }
      runtimeCallResults =
        createRuntimeCall(rewriter, loc, "mapMake", this->getTypeConverter(), args);
    }
    else if (mlir::isa<InterfaceType>(baseType))
    {
      SmallVector<Value> args = adaptor.getOperands();
      runtimeCallResults =
        createRuntimeCall(rewriter, loc, "interfaceMake", this->getTypeConverter(), args);
    }
    else
    {
      return failure();
    }

    rewriter.replaceOp(op, runtimeCallResults);
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

struct PanicOpLowering : ConvertOpToLLVMPattern<PanicOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(PanicOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    const auto loc = op.getLoc();

    // Create the runtime call to schedule this function call
    createRuntimeCall(rewriter, loc, "_panic", {}, { adaptor.getValue() });

    // The panic operation may or may not branch to the parent function's recover block if it
    // exists.
    if (op->hasSuccessors())
    {
      // Branch to the recover block
      rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, op->getSuccessor(0));
    }
    else
    {
      SmallVector<Type> resultTypes;
      if (failed(this->getTypeConverter()->convertTypes(
            op->getParentOp()->getResultTypes(), resultTypes)))
      {
        return failure();
      }

      // The end of the function should be unreachable
      rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, resultTypes);
    }
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
    const auto runtimeCallResults = createRuntimeCall(rewriter, loc, "_recover", {}, {});
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

struct RuntimeCallOpLowering : ConvertOpToLLVMPattern<RuntimeCallOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    RuntimeCallOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    Type packedResult = nullptr;
    unsigned numResults = op.getNumResults();
    auto resultTypes = llvm::to_vector<4>(op.getResultTypes());
    auto useBarePtrCallConv = getTypeConverter()->getOptions().useBarePtrCallConv;

    if (numResults != 0)
    {
      packedResult = this->getTypeConverter()->packFunctionResults(resultTypes, useBarePtrCallConv);
      if (!packedResult)
      {
        return failure();
      }
    }

    auto callOp = rewriter.create<mlir::LLVM::CallOp>(
      op.getLoc(),
      packedResult ? TypeRange(packedResult) : TypeRange(),
      adaptor.getCalleeOperands(),
      op->getAttrs());

    SmallVector<Value, 4> results;
    if (numResults < 2)
    {
      // Return directly
      results.append(callOp.result_begin(), callOp.result_end());
    }
    else
    {
      // Unpack result struct
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i)
      {
        results.push_back(
          rewriter.create<mlir::LLVM::ExtractValueOp>(callOp.getLoc(), callOp->getResult(0), i));
      }
    }
    rewriter.replaceOp(op, results);
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
      op, value, addr, alignment, isVolatile, false, ordering);
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

struct TypeInfoOpLowering : ConvertOpToLLVMPattern<TypeInfoOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    TypeInfoOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    auto module = op->getParentOfType<ModuleOp>();
    auto typeInfoGlobalOp = createTypeInfo(rewriter, module, op.getLoc(), op.getT());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, typeInfoGlobalOp);
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
            transforms::LLVM::DeferOpLowering,
            transforms::LLVM::ExtractOpLowering,
            transforms::LLVM::GetElementPointerOpLowering,
            transforms::LLVM::GlobalOpLowering,
            transforms::LLVM::GlobalCtorsOpLowering,
            transforms::LLVM::InsertOpLowering,
            transforms::LLVM::InterfaceCallOpLowering,
            transforms::LLVM::IntToPtrOpLowering,
            transforms::LLVM::LoadOpLowering,
            transforms::LLVM::MakeOpLowering,
            transforms::LLVM::MakeInterfaceOpLowering,
            transforms::LLVM::PanicOpLowering,
            transforms::LLVM::PointerToFunctionOpLowering,
            transforms::LLVM::PtrToIntOpLowering,
            transforms::LLVM::RecoverOpLowering,
            transforms::LLVM::RecvOpLowering,
            transforms::LLVM::RuntimeCallOpLowering,
            transforms::LLVM::StoreOpLowering,
            transforms::LLVM::TypeInfoOpLowering,
            transforms::LLVM::YieldOpLowering,
            transforms::LLVM::ZeroOpLowering
        >(converter);
        // clang-format off
    }
} // namespace mlir::go
