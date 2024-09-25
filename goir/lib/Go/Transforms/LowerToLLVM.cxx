
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

static SmallVector<Value> createRuntimeCall(
  PatternRewriter& rewriter,
  const Location location,
  const std::string& funcName,
  const ArrayRef<Type>& results,
  const ArrayRef<Value>& args)
{
  // Format the fully qualified function name
  const std::string qualifiedFuncName = "runtime." + funcName;
  const auto callee = FlatSymbolRefAttr::get(rewriter.getContext(), qualifiedFuncName);
  const auto numResults = results.size();
  SmallVector<Value, 4> resultValues;

  // Build the expected function signature.
  Type returnType;
  if (numResults == 0)
  {
    returnType = LLVM::LLVMVoidType::get(rewriter.getContext());
  }
  else if (results.size() == 1)
  {
    returnType = results[0];
  }
  else if (results.size() > 1)
  {
    returnType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), results, false);
  }

  SmallVector<Type> argTypes(args.size());
  for (size_t i = 0; i < args.size(); ++i)
  {
    argTypes[i] = args[i].getType();
  }

  auto fnType = LLVM::LLVMFunctionType::get(rewriter.getContext(), returnType, argTypes, false);

  // Create the call.
  auto callOp = rewriter.create<LLVM::CallOp>(location, fnType, callee, args);

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
    auto type = typeConverter->convertType(op.getType());
    auto runtimeCallResults = createRuntimeCall(
      rewriter, loc, "stringConcat", { type }, { adaptor.getLhs(), adaptor.getRhs() });
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
    auto elementType = this->typeConverter->convertType(adaptor.getElement());
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
        createRuntimeCall(rewriter, loc, "alloc", { getVoidPtrType() }, { sizeValue });
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
    auto resultType = typeConverter->convertType(op.getType());
    if (
      mlir::isa<mlir::LLVM::LLVMStructType>(resultType) ||
      adaptor.getValue().getType() == resultType)
    {
      // Primitive to runtime type conversion.
      rewriter.replaceOp(op, adaptor.getValue());
    }
    else
    {
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, resultType, adaptor.getValue());
    }
    return success();
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
    const auto intType = mlir::IntegerType::get(this->getContext(), this->getTypeConverter()->getPointerBitwidth());
    const auto ptrType = mlir::LLVM::LLVMPointerType::get(this->getContext());
    const auto boolType = mlir::IntegerType::get(this->getContext(), 1);
    mlir::DataLayout dataLayout(module);

    SmallVector<Type> resultTypes;
    if (failed(this->typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
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
        resultTypes,
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
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "channelCap", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "sliceCap", resultTypes, { operands[0] });
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
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "mapClear", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "sliceClear", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          });
    }
    else if (callee == "close")
    {
      createRuntimeCall(rewriter, loc, "channelClose", resultTypes, { operands[0] });
      rewriter.eraseOp(op);
    }
    else if (callee == "copy")
    {
      const auto srcType = op.getOperandTypes()[1];
      llvm::TypeSwitch<Type>(srcType)
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "sliceCopy", resultTypes, { operands[0], operands[1] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](StringType type)
          {
            const auto runtimeCallResults = createRuntimeCall(
              rewriter, loc, "sliceCopyString", resultTypes, { operands[0], operands[1] });
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
      createRuntimeCall(rewriter, loc, "mapDelete", resultTypes, { operands[0], keyValue });
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
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "channelLen", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](MapType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "mapLen", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](SliceType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "sliceLen", resultTypes, { operands[0] });
            rewriter.replaceOp(op, runtimeCallResults);
          })
        .Case(
          [&](StringType type)
          {
            const auto runtimeCallResults =
              createRuntimeCall(rewriter, loc, "stringLen", resultTypes, { operands[0] });
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
              createRuntimeCall(rewriter, loc, "channelMake", resultTypes, args);
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
              createRuntimeCall(rewriter, loc, "mapMake", resultTypes, args);
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
              createRuntimeCall(rewriter, loc, "sliceMake", resultTypes, args);
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
      createRuntimeCall(rewriter, loc, "_panic", resultTypes, { operands[0] });
      rewriter.eraseOp(op);
    }
    else if (callee == "print")
    {
      createRuntimeCall(rewriter, loc, "_print", resultTypes, SmallVector<Value>(operands));
      rewriter.eraseOp(op);
    }
    else if (callee == "println")
    {
      createRuntimeCall(rewriter, loc, "_println", resultTypes, SmallVector<Value>(operands));
      rewriter.eraseOp(op);
    }
    else if (callee == "recover")
    {
      const auto runtimeCallResults = createRuntimeCall(rewriter, loc, "_recover", resultTypes, {});
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
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "slice", resultTypes, { operands[0], operands[1] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.SliceData")
    {
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "sliceData", resultTypes, { operands[0] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.String")
    {
      const auto runtimeCallResults = createRuntimeCall(
        rewriter, loc, "stringFromPointer", resultTypes, { operands[0], operands[1] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else if (callee == "unsafe.StringData")
    {
      const auto runtimeCallResults =
        createRuntimeCall(rewriter, loc, "stringData", resultTypes, { operands[0] });
      rewriter.replaceOp(op, runtimeCallResults);
    }
    else
    {
      return failure();
    }
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
    auto resultType = this->typeConverter->convertType(op.getType());

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

struct ConstantOpLowering : ConvertOpToLLVMPattern<ConstantOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
    ConstantOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const override
  {
    const auto loc = op.getLoc();
    auto resultType = this->typeConverter->convertType(op.getType());

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

struct ZeroOpLowering : ConvertOpToLLVMPattern<ZeroOp>
{
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(ZeroOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    auto resultType = this->typeConverter->convertType(op.getType());
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
      rewriter,
      loc,
      "alloc",
      { mlir::LLVM::LLVMPointerType::get(rewriter.getContext()) },
      { constantIntOp.getResult() });

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

    auto elemT = this->typeConverter->convertType(adaptor.getGlobalType());

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
      argTypes.push_back(this->typeConverter->convertType(arg.getType()));
    }

    for (auto result : op->getResults())
    {
      resultTypes.push_back(this->typeConverter->convertType(result.getType()));
    }

    // Create a constant int value for the hash
    auto constantIntOp =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), methodHash);
    auto constHashValue = constantIntOp.getResult();

    // Perform vtable lookup
    auto runtimeCallResults = createRuntimeCall(
      rewriter, loc, "interfaceLookUp", { ptrType, ptrType }, { ifaceValue, constHashValue });

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
    auto type = typeConverter->convertType(op.getType());
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
      runtimeCallResults = createRuntimeCall(rewriter, loc, "chanMake", type, args);
    }
    else if (mlir::isa<SliceType>(baseType))
    {
      SmallVector<Value> args = adaptor.getOperands();
      runtimeCallResults = createRuntimeCall(rewriter, loc, "sliceMake", type, args);
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
      runtimeCallResults = createRuntimeCall(rewriter, loc, "mapMake", type, args);
    }
    else if (mlir::isa<InterfaceType>(baseType))
    {
      SmallVector<Value> args = adaptor.getOperands();
      runtimeCallResults = createRuntimeCall(rewriter, loc, "interfaceMake", type, args);
    }
    else
    {
      return failure();
    }

    rewriter.replaceOp(op, runtimeCallResults);
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
      if (failed(
            this->typeConverter->convertTypes(op->getParentOp()->getResultTypes(), resultTypes)))
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
    auto arrSizeConstOp =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(rewriter.getContext(), 64), 1);
    auto allocaOp = rewriter.create<mlir::LLVM::AllocaOp>(loc, type, arrSizeConstOp.getResult());

    // Create the runtime call to receive a value from the channel
    auto blockConstOp = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerType::get(rewriter.getContext(), 64), adaptor.getCommaOk() ? 1 : 0);
    createRuntimeCall(
      rewriter,
      loc,
      "_channelReceive",
      { mlir::IntegerType::get(rewriter.getContext(), 1) },
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
            transforms::LLVM::ChangeInterfaceOpLowering,
            transforms::LLVM::ConstantOpLowering,
            transforms::LLVM::DeferOpLowering,
            transforms::LLVM::GetElementPointerOpLowering,
            transforms::LLVM::GlobalOpLowering,
            transforms::LLVM::GlobalCtorsOpLowering,
            transforms::LLVM::InterfaceCallOpLowering,
            transforms::LLVM::IntToPtrOpLowering,
            transforms::LLVM::LoadOpLowering,
            transforms::LLVM::MakeOpLowering,
            transforms::LLVM::PanicOpLowering,
            transforms::LLVM::PointerToFunctionOpLowering,
            transforms::LLVM::PtrToIntOpLowering,
            transforms::LLVM::RecoverOpLowering,
            transforms::LLVM::RecvOpLowering,
            transforms::LLVM::RuntimeCallOpLowering,
            transforms::LLVM::StoreOpLowering,
            transforms::LLVM::ExtractOpLowering,
            transforms::LLVM::InsertOpLowering,
            transforms::LLVM::TypeInfoOpLowering,
            transforms::LLVM::YieldOpLowering,
            transforms::LLVM::ZeroOpLowering
        >(converter);
        // clang-format off
    }
} // namespace mlir::go
