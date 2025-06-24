#include <limits>

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

constexpr std::string_view runtimeFuncTypeName = "runtime._func";

namespace mlir::go
{

template<typename resultT, typename opT, typename adaptorT>
std::optional<resultT> getOrFold(opT op, adaptorT adaptor, const size_t index)
{
  resultT operand;
  if (const auto attr = adaptor.getCalleeOperands()[index])
  {
    operand = mlir::dyn_cast_or_null<resultT>(attr);
  }
  else if (auto definingOp = op->getCalleeOperands()[index].getDefiningOp())
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (failed(definingOp->fold(results)))
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

::mlir::LogicalResult DeferOp::verify()
{
  FunctionType Fn;

  // Validate attribute combinations.
  const bool hasIface = !!this->getIfaceValue();
  const bool hasSym = this->getSymName().has_value();
  const bool hasCallee = !!this->getCalleeValue();

  const int calleeCount =
    static_cast<int>(hasIface) + static_cast<int>(hasSym) + static_cast<int>(hasCallee);
  if (calleeCount != 1)
  {
    return this->emitOpError()
      << "must have exactly one of callee_value, iface_value, or sym_name set";
  }

  if (hasSym && (hasIface || hasCallee))
  {
    return this->emitOpError()
      << "sym_name cannot be specified together with callee_value or iface_value";
  }

  if (hasIface && !this->getMethodName())
  {
    return this->emitOpError() << "method_name must be specified when iface_value is used";
  }

  if (hasIface)
  {
    auto ifaceType = mlir::go::dyn_cast<InterfaceType>(this->getIfaceValue().getType());
    if (!ifaceType)
    {
      return this->emitOpError() << "iface_value must have an interface type";
    }

    auto methods = ifaceType.getMethods();
    auto it = methods.find(this->getMethodNameAttr().str());
    if (it == methods.end())
    {
      return this->emitOpError() << "method \"" << this->getMethodNameAttr()
                                 << "\" not found in interface";
    }

    Fn = mlir::cast<mlir::go::FunctionType>(it->second);
  }

  if (hasSym)
  {
    auto moduleOp = this->getOperation()->getParentOfType<mlir::ModuleOp>();
    auto funcOp = moduleOp.lookupSymbol(*this->getSymName());
    if (!funcOp)
    {
      return this->emitOpError() << "callee with symbol \"" << *this->getSymName()
                                 << "\" not found in module";
    }

    // TODO: GoOp is lowered in a later LLVM lowering pass. So, the following check will fail when
    //       functions are lowered to func.func.

    auto goFuncOp = mlir::dyn_cast<mlir::go::FuncOp>(funcOp);
    if (!goFuncOp)
    {
      // TODO: For now, no further verification can take place.
      return success();

      // return this->emitOpError() << "callee symbol must reference a mlir::go::FuncOp, but got "
      //                            << funcOp;
    }

    Fn = mlir::go::dyn_cast<mlir::go::FunctionType>(goFuncOp.getFunctionType());
  }

  if (hasCallee)
  {
    // TODO: support passing a signature with the callee value.
    // Skipping function signature validation for now.
    return success();
  }

  // Validate operands against function signature.
  const auto numOperands = this->getCalleeOperands().size();
  const auto expectedNumArgs =
    Fn.hasReceiver() ? (hasSym ? Fn.getNumInputs() + 1 : Fn.getNumInputs()) : Fn.getNumInputs();
  if (numOperands != expectedNumArgs)
  {
    return this->emitOpError() << "expected " << expectedNumArgs
                               << " operands for function call, but got " << numOperands;
  }

  // Validate receiver type.
  size_t operandIdx = 0;
  if (Fn.hasReceiver())
  {
    auto got = this->getCalleeOperands()[0].getType();
    auto expected = Fn.getReceiver();
    if (got != expected)
    {
      return this->emitOpError() << "expected receiver type " << expected << ", but got " << got;
    }
    operandIdx = 1;
  }

  // Validate parameter types.
  for (size_t i = 0; i < Fn.getNumInputs(); ++i, ++operandIdx)
  {
    auto got = this->getCalleeOperands()[operandIdx].getType();
    auto expected = Fn.getInput(i);
    if (got != expected)
    {
      return this->emitOpError() << "expected type " << expected << " for parameter " << i
                                 << ", but got " << got;
    }
  }
  return success();
}

::mlir::LogicalResult GoOp::verify()
{
  FunctionType Fn;

  // Validate attribute combinations.
  const bool hasIface = !!this->getIfaceValue();
  const bool hasSym = this->getSymName().has_value();
  const bool hasCallee = !!this->getCalleeValue();

  const int calleeCount =
    static_cast<int>(hasIface) + static_cast<int>(hasSym) + static_cast<int>(hasCallee);
  if (calleeCount != 1)
  {
    return this->emitOpError()
      << "must have exactly one of callee_value, iface_value, or sym_name set";
  }

  if (hasSym && (hasIface || hasCallee))
  {
    return this->emitOpError()
      << "sym_name cannot be specified together with callee_value or iface_value";
  }

  if (hasIface && !this->getMethodName())
  {
    return this->emitOpError() << "method_name must be specified when iface_value is used";
  }

  if (hasIface)
  {
    auto ifaceType = mlir::go::dyn_cast<InterfaceType>(this->getIfaceValue().getType());
    if (!ifaceType)
    {
      return this->emitOpError() << "iface_value must have an interface type";
    }

    auto methods = ifaceType.getMethods();
    auto it = methods.find(this->getMethodNameAttr().str());
    if (it == methods.end())
    {
      return this->emitOpError() << "method \"" << this->getMethodNameAttr()
                                 << "\" not found in interface";
    }

    Fn = mlir::cast<mlir::go::FunctionType>(it->second);
  }

  if (hasSym)
  {
    auto moduleOp = this->getOperation()->getParentOfType<mlir::ModuleOp>();
    auto funcOp = moduleOp.lookupSymbol(*this->getSymName());
    if (!funcOp)
    {
      return this->emitOpError() << "callee with symbol \"" << *this->getSymName()
                                 << "\" not found in module";
    }

    // TODO: GoOp is lowered in a later LLVM lowering pass. So, the following check will fail when
    //       functions are lowered to func.func.

    auto goFuncOp = mlir::dyn_cast<mlir::go::FuncOp>(funcOp);
    if (!goFuncOp)
    {
      // TODO: For now, no further verification can take place.
      return success();

      // return this->emitOpError() << "callee symbol must reference a mlir::go::FuncOp, but got "
      //                            << funcOp;
    }

    Fn = mlir::go::dyn_cast<mlir::go::FunctionType>(goFuncOp.getFunctionType());
  }

  if (hasCallee)
  {
    // TODO: support passing a signature with the callee value.
    // Skipping function signature validation for now.
    return success();
  }

  // Validate operands against function signature.
  const auto numOperands = this->getCalleeOperands().size();
  const auto expectedNumArgs =
    Fn.hasReceiver() ? (hasSym ? Fn.getNumInputs() + 1 : Fn.getNumInputs()) : Fn.getNumInputs();
  if (numOperands != expectedNumArgs)
  {
    return this->emitOpError() << "expected " << expectedNumArgs
                               << " operands for function call, but got " << numOperands;
  }

  // Validate receiver type.
  size_t operandIdx = 0;
  if (Fn.hasReceiver())
  {
    auto got = this->getCalleeOperands()[0].getType();
    auto expected = Fn.getReceiver();
    if (got != expected)
    {
      return this->emitOpError() << "expected receiver type " << expected << ", but got " << got;
    }
    operandIdx = 1;
  }

  // Validate parameter types.
  for (size_t i = 0; i < Fn.getNumInputs(); ++i, ++operandIdx)
  {
    auto got = this->getCalleeOperands()[operandIdx].getType();
    auto expected = Fn.getInput(i);
    if (got != expected)
    {
      return this->emitOpError() << "expected type " << expected << " for parameter " << i
      << ", but got " << got;
    }
  }
  return success();
}

mlir::LogicalResult InterfaceCallOp::verify()
{
  auto type = cast<InterfaceType>(this->getIface().getType());
  const auto methods = type.getMethods();
  const auto args = this->getCalleeOperands();
  const auto results = this->getResultTypes();

  // The callee cannot be an empty string
  if (this->getCallee().empty())
  {
    return this->emitOpError() << "callee cannot be an empty string";
  }

  // The callee must be a method defined in the interface
  auto it = methods.find(this->getCallee().str());
  if (it == methods.cend())
  {
    return this->emitOpError() << "callee does not exist in the specified interface type";
  }

  const auto fnT = mlir::cast<FunctionType>(it->second);

  // The call arguments, excluding the receiver, must match
  // Fast-path
  if (args.size() != fnT.getNumInputs())
  {
    return this->emitOpError() << "mismatch in number of inputs vs. method signature";
  }

  if (results.size() != fnT.getNumResults())
  {
    return this->emitOpError() << "mismatch in number of results vs. method signature";
  }

  // Match the call arguments
  for (size_t i = 1; i < args.size(); ++i)
  {
    if (args[i].getType() != fnT.getInput(i))
    {
      return this->emitOpError() << "argument type " << i << " does not match signature";
    }
  }

  // Match the call results
  for (size_t i = 0; i < results.size(); ++i)
  {
    if (results[i] != fnT.getResult(i))
    {
      return this->emitOpError() << "result type " << i << " does not match signature";
    }
  }

  return success();
}

LogicalResult BuiltInCallOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& results)
{
  if (adaptor.getOperands().empty())
  {
    return failure();
  }

  const auto i64Type = mlir::IntegerType::get(this->getContext(), 64);
  const auto f64Type = mlir::Float64Type::get(this->getContext());
  const auto callee = this->getCallee().str();
  if (callee == "cap")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](ArrayType type)
        {
          const auto result = mlir::IntegerAttr::get(i64Type, type.getLength());
          results.push_back(result);
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "len")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](ArrayType type)
        {
          const auto result = mlir::IntegerAttr::get(i64Type, type.getLength());
          results.push_back(result);
          return success();
        })
      .Case(
        [&](StringType type)
        {
          const auto value = getOrFold<StringAttr>(this, adaptor, 0).value_or(StringAttr());
          if (!value)
          {
            return failure();
          }

          // Return the length of the constant string.
          const auto result = mlir::IntegerAttr::get(i64Type, value.size());
          results.push_back(result);
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "imag")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](ComplexType type)
        {
          const auto value =
            getOrFold<ComplexNumberAttr>(this, adaptor, 0).value_or(ComplexNumberAttr());
          if (!value)
          {
            return failure();
          }
          results.push_back(value.getImag());
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "real")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](ComplexType type)
        {
          const auto value =
            getOrFold<ComplexNumberAttr>(this, adaptor, 0).value_or(ComplexNumberAttr());
          if (!value)
          {
            return failure();
          }
          results.push_back(value.getReal());
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "max")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](IntegerType)
        {
          int64_t largestValue = std::numeric_limits<int64_t>::min();
          for (const auto operand : llvm::seq(this->getCalleeOperands().size()))
          {
            const auto value =
              getOrFold<IntegerAttr>(this, adaptor, operand).value_or(IntegerAttr());
            if (!value)
            {
              return failure();
            }

            if (value.getInt() > largestValue)
            {
              largestValue = value.getInt();
            }
          }
          const auto result = mlir::IntegerAttr::get(i64Type, largestValue);
          results.push_back(result);
          return success();
        })
      .Case(
        [&](FloatType)
        {
          double largestValue = std::numeric_limits<double>::min();
          for (const auto operand : llvm::seq(this->getCalleeOperands().size()))
          {
            const auto value = getOrFold<FloatAttr>(this, adaptor, operand).value_or(FloatAttr());
            if (!value)
            {
              return failure();
            }

            if (value.getValueAsDouble() > largestValue)
            {
              largestValue = value.getValueAsDouble();
            }
          }
          const auto result = mlir::FloatAttr::get(f64Type, largestValue);
          results.push_back(result);
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "min")
  {
    const auto inputType = this->getCalleeOperands()[0].getType();
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(inputType)
      .Case(
        [&](IntegerType)
        {
          int64_t smallestValue = std::numeric_limits<int64_t>::max();
          for (const auto operand : llvm::seq(this->getCalleeOperands().size()))
          {
            const auto value =
              getOrFold<IntegerAttr>(this, adaptor, operand).value_or(IntegerAttr());
            if (!value)
            {
              return failure();
            }

            if (value.getInt() < smallestValue)
            {
              smallestValue = value.getInt();
            }
          }
          const auto result = mlir::IntegerAttr::get(i64Type, smallestValue);
          results.push_back(result);
          return success();
        })
      .Case(
        [&](FloatType)
        {
          double smallestValue = std::numeric_limits<double>::max();
          for (const auto operand : llvm::seq(this->getCalleeOperands().size()))
          {
            const auto value = getOrFold<FloatAttr>(this, adaptor, operand).value_or(FloatAttr());
            if (!value)
            {
              return failure();
            }

            if (value.getValueAsDouble() < smallestValue)
            {
              smallestValue = value.getValueAsDouble();
            }
          }
          const auto result = mlir::FloatAttr::get(f64Type, smallestValue);
          results.push_back(result);
          return success();
        })
      .Default([&](Type) { return failure(); });
  }

  if (callee == "unsafe.Add")
  {
    int64_t sum = 0;
    for (const auto operand : llvm::seq(this->getCalleeOperands().size()))
    {
      const auto value = getOrFold<IntegerAttr>(this, adaptor, operand).value_or(IntegerAttr());
      if (!value)
      {
        return failure();
      }

      sum += value.getInt();
    }
    const auto result = mlir::IntegerAttr::get(i64Type, sum);
    results.push_back(result);
    return success();
  }

  if (callee == "unsafe.Alignof")
  {
    const auto dataLayout = DataLayout(getOperation()->getParentOfType<ModuleOp>());
    const auto size = dataLayout.getTypeABIAlignment(this->getCalleeOperands()[0].getType());
    const auto result = IntegerAttr::get(i64Type, size);
    results.push_back(result);
    return success();
  }

  if (callee == "unsafe.Offsetof")
  {
    const auto structType =
      mlir::dyn_cast_or_null<GoStructType>(this->getCalleeOperands()[0].getType());
    if (!structType)
    {
      return failure();
    }

    const auto index = mlir::dyn_cast_or_null<IntegerAttr>(adaptor.getCalleeOperands()[1]);
    if (!index)
    {
      return failure();
    }

    const auto dataLayout = DataLayout(getOperation()->getParentOfType<ModuleOp>());
    const auto size = structType.getFieldOffset(dataLayout, index.getInt());
    const auto result = IntegerAttr::get(i64Type, size);
    results.push_back(result);
    return success();
  }

  if (callee == "unsafe.Sizeof")
  {
    const auto type = this->getCalleeOperands()[0].getType();
    const auto dataLayout = DataLayout(getOperation()->getParentOfType<ModuleOp>());
    const auto size = dataLayout.getTypeSize(type);
    const auto result = IntegerAttr::get(i64Type, size);
    results.push_back(result);
    return success();
  }

  return failure();
}

mlir::LogicalResult BuiltInCallOp::verify()
{
  const auto intType = IntegerType::get(this->getContext(), IntegerType::Signed);
  const auto charType = IntegerType::get(this->getContext(), IntegerType::Unsigned, 8);

  const auto callee = this->getCallee().str();
  if (callee == "append")
  {
    // Must have EXACTLY 2 arguments to be valid.
    if (this->getNumOperands() != 2)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected at least 2 operands";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // First operand MUST be a slice.
    if (!go::isCompatibleType<SliceType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected slice but got " << this->getOperand(0).getType()
                                 << "for operand 0";
    }

    const auto sliceType = go::cast<SliceType>(this->getOperand(0).getType());

    // Second operand MUST be another slice whose element type matches that of the input slice.
    if (this->getOperand(1).getType() != sliceType)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected type " << sliceType << " for operand 1, but got "
                                 << this->getOperand(0).getType();
    }

    // Result type must match the input slice's type.
    if (this->getResult(0).getType() != sliceType)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result type " << sliceType << " but got "
                                 << this->getResult(0).getType();
    }
  }
  else if (callee == "cap")
  {
    // Must have EXACTLY 1 arguments to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // Input value MUST be a chan, map or slice.
    const auto inputType = this->getOperand(0).getType();
    if (
      !go::isCompatibleType<ArrayType>(inputType) && !go::isCompatibleType<ChanType>(inputType) &&
      !go::isCompatibleType<SliceType>(inputType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected input of type array, chan or slice. Got "
                                 << inputType;
    }

    // The result type MUST be an integer.
    if (this->getResult(0).getType() != intType)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result of int (!go.int) type but got "
                                 << this->getResult(0).getType();
    }
  }
  else if (callee == "clear")
  {
    // Must have EXACTLY 1 arguments to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ZERO values.
    if (this->getNumResults() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 results";
    }

    // Input value MUST be a map or a slice.
    const auto inputType = this->getOperand(0).getType();
    if (!go::isCompatibleType<MapType>(inputType) && !go::isCompatibleType<SliceType>(inputType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected input of type map or slice. Got " << inputType;
    }
  }
  else if (callee == "close")
  {
    // Must have EXACTLY 1 arguments to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ZERO values.
    if (this->getNumResults() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 results";
    }

    // Input value MUST be an array, chan, map, slice or string.
    const auto inputType = this->getOperand(0).getType();
    if (!go::isCompatibleType<ChanType>(inputType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected a chan input type. Got " << inputType;
    }
  }
  else if (callee == "complex")
  {
    // Must have EXACTLY 2 arguments to be valid.
    if (this->getNumOperands() != 2)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // The input operands MUST be floating-point values.
    if (!go::isCompatibleType<FloatType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected floating-point operand type for operand 0 but got "
                                 << this->getOperand(0).getType();
    }

    if (!go::isCompatibleType<FloatType>(this->getOperand(1).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected floating-point operand type for operand 1 but got "
                                 << this->getOperand(1).getType();
    }

    // The result MUST be a complex number type.
    if (!go::isCompatibleType<ComplexType>(this->getResult(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected a floating-point result type but got "
                                 << this->getResult(0).getType();
    }

    const auto operand0Type = go::cast<FloatType>(this->getOperand(0).getType());
    const auto operand1Type = go::cast<FloatType>(this->getOperand(1).getType());
    const auto resultType = go::cast<ComplexType>(this->getResult(0).getType());

    // The operand types MUST match.
    if (operand0Type != operand1Type)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected operand types to match, but got " << operand0Type
                                 << " and " << operand1Type;
    }

    // The bit-width of the resulting float must match that of the respective complex number type.
    if (operand0Type.getIntOrFloatBitWidth() != resultType.getIntOrFloatBitWidth())
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result bit-width of "
                                 << operand0Type.getIntOrFloatBitWidth() << " but got "
                                 << resultType;
    }
  }
  else if (callee == "copy")
  {
    // Must have EXACTLY 2 arguments to be valid.
    if (this->getNumOperands() != 2)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected at least 2 operands";
    }

    // Must return exactly 1 value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << "expected 1 result";
    }

    // The result type MUST be an integer.
    if (this->getResult(0).getType() != intType)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected integer result type but got "
                                 << this->getResult(0).getType();
    }

    // The first operand MUST be a slice.
    if (!go::isCompatibleType<SliceType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected slice but got " << this->getOperand(0).getType()
                                 << "for operand 0";
    }

    const auto inputSliceType = go::cast<SliceType>(this->getOperand(0).getType());

    // Verify based on the type of the second operand.
    return TypeSwitch<Type, LogicalResult>(this->getOperand(1).getType())
      .Case(
        [&](StringType stringType) -> LogicalResult
        {
          // The input slice type MUST be a byte slice.
          if (inputSliceType.getElementType() != charType)
          {
            return this->emitOpError() << callee << ": "
                                       << "expected byte slice ([]byte) but got "
                                       << this->getOperand(0).getType() << "for operand 0";
          }
          return success();
        })
      .Case(
        [&](SliceType sliceType) -> LogicalResult
        {
          // Both must have the same type.
          if (inputSliceType != this->getOperand(1).getType())
          {
            return this->emitOpError()
              << callee << ": "
              << "expected " << this->getOperand(1).getType() << " but got "
              << this->getOperand(0).getType() << "for operand 0";
          }
          return success();
        })
      .Default(
        [&](Type type) -> LogicalResult
        {
          return this->emitOpError() << callee << ": "
                                     << "expected slice type or string type but got "
                                     << this->getOperand(1).getType() << "for operand 1";
        });
  }
  else if (callee == "delete")
  {
    // Must have EXACTLY 1 arguments to be valid.
    if (this->getNumOperands() != 2)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 2 operands";
    }

    // Must return exactly ZERO values.
    if (this->getNumResults() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 results";
    }

    // Input value MUST be a map.
    const auto inputType = this->getOperand(0).getType();
    if (!go::isCompatibleType<MapType>(inputType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected map operand type. Got " << inputType;
    }
  }
  else if (callee == "imag" || callee == "real")
  {
    // Must have EXACTLY 1 arguments to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // The input operand MUST be a complex number.
    if (!go::isCompatibleType<ComplexType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected complex operand type but got "
                                 << this->getOperand(0).getType();
    }

    // The result MUST be a float type.
    if (!go::isCompatibleType<FloatType>(this->getResult(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected a floating-point result type but got "
                                 << this->getResult(0).getType();
    }

    // The bit-width of the resulting float must match that of the respective complex number type.
    const auto inputType = go::cast<ComplexType>(this->getOperand(0).getType());
    const auto resultType = go::cast<FloatType>(this->getResult(0).getType());
    if (inputType.getIntOrFloatBitWidth() != resultType.getIntOrFloatBitWidth())
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result bit-width of "
                                 << inputType.getIntOrFloatBitWidth() << " but got " << resultType;
    }
  }
  else if (callee == "len")
  {
    // Must have EXACTLY 1 operand to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // Input value MUST be an array, chan, map, slice or string.
    const auto inputType = this->getOperand(0).getType();
    if (
      !go::isCompatibleType<ArrayType>(inputType) && !go::isCompatibleType<ChanType>(inputType) &&
      !go::isCompatibleType<MapType>(inputType) && !go::isCompatibleType<SliceType>(inputType) &&
      !go::isCompatibleType<StringType>(inputType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected input of type array, chan, map, slice or string. Got "
                                 << inputType;
    }

    // The result type MUST be an integer.
    if (this->getResult(0).getType() != intType)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result of int (!go.int) type but got "
                                 << this->getResult(0).getType();
    }
  }
  else if (callee == "make")
  {
    // Mus return EXACTLY a single result.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 result";
    }

    // Verify based on the return type.
    return TypeSwitch<Type, LogicalResult>(this->getResult(0).getType())
      .Case(
        [&](ChanType chanType) -> LogicalResult
        {
          // Must have AT MOST one operand to be valid.
          if (this->getNumOperands() > 1)
          {
            return this->emitOpError()
              << callee << ": "
              << "expected at most 1 operand, but got " << this->getNumOperands() << " operands";
          }

          if (this->getNumOperands() == 1)
          {
            const auto operandType = this->getOperand(0).getType();
            // The operand must be an integer type.
            if (!isIntegerType(operandType))
            {
              return this->emitOpError() << callee << ": "
                                         << "expected operand type of int but got " << operandType;
            }
          }
          return success();
        })
      .Case(
        [&](MapType mapType) -> LogicalResult
        {
          // Must have AT MOST one operand to be valid.
          if (this->getNumOperands() > 1)
          {
            return this->emitOpError()
              << callee << ": "
              << "expected at most 1 operand, but got " << this->getNumOperands() << " operands";
          }

          if (this->getNumOperands() == 1)
          {
            const auto operandType = this->getOperand(0).getType();
            // The operand must be an integer type.
            if (!isIntegerType(operandType))
            {
              return this->emitOpError()
                << callee << ": "
                << "expected operand type of integer but got " << operandType;
            }
          }
          return success();
        })
      .Case(
        [&](SliceType sliceType) -> LogicalResult
        {
          // Must have AT MOST two operand to be valid.
          if (this->getNumOperands() > 2)
          {
            return this->emitOpError()
              << callee << ": "
              << "expected at most 1 operand, but got " << this->getNumOperands() << " operands";
          }

          // Operands MUST be integers.
          for (size_t i = 0; i < this->getNumOperands(); i++)
          {
            const auto operandType = this->getOperand(i).getType();
            if (!isIntegerType(operandType))
            {
              return this->emitOpError() << callee << ": "
                                         << "expected operand type of integer for operand " << i
                                         << " but got " << operandType;
            }
          }
          return success();
        })
      .Default(
        [&](Type type) -> LogicalResult
        {
          return this->emitOpError()
            << callee << ": "
            << "expected result of type chan, map or slice, but got " << type;
        });
    // return this->emitOpError() << "use respective make operation";
  }
  else if (callee == "max" || callee == "min")
  {
    // Must have at least 1 operand to be valid.
    if (this->getNumOperands() < 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected at least 1 operand";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // Operands must be ordered types.
    for (size_t i = 0; i < this->getNumOperands(); ++i)
    {
      const auto operandType = this->getOperand(i).getType();
      if (!isOrderedType(operandType))
      {
        return this->emitOpError()
          << callee << ": "
          << "expected integer type, floating-point type or string type for operand " << i
          << " but got " << operandType;
      }
    }
  }
  else if (callee == "new")
  {
    // Must have EXACTLY 0 operands to be valid.
    if (this->getNumOperands() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 0 operands";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // Result type MUST be a pointer.
    const auto resultType = this->getResultTypes()[0];
    if (!go::isCompatibleType<PointerType>(resultType))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected result of a pointer type but got " << resultType;
    }

    //  Result CANNOT be an unsafe pointer.
    if (isUnsafePointer(this->getResult(0).getType()))
    {
      return this->emitOpError()
        << callee << ": "
        << "expected result of a pointer type with a valid base type but got "
        << this->getResult(0).getType();
    }
  }
  else if (callee == "panic")
  {
    // Must have EXACTLY 1 operand to be valid.
    if (this->getNumOperands() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 operand";
    }

    if (this->getNumOperands() == 1)
    {
      const auto operandType = this->getOperand(0).getType();
      // The operand must be an interface type.
      if (!go::isAnyType(operandType))
      {
        return this->emitOpError()
          << callee << ": "
          << "expected operand of interface (any) type but got " << operandType;
      }
    }

    // Must return exactly ZERO values.
    if (this->getNumResults() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 results";
    }
  }
  else if (callee == "print" || callee == "println")
  {
    // Must return exactly ZERO results.
    if (this->getNumResults() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 results";
    }
  }
  else if (callee == "recover")
  {
    // Must have exactly ZERO operands.
    if (this->getNumOperands() != 0)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 0 operands";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 1 result";
    }

    // Result must be interface (any) type
    if (this->getNumResults() == 1)
    {
      const auto resultType = this->getResult(0).getType();
      // The operand must be an interface type.
      if (!go::isAnyType(resultType))
      {
        return this->emitOpError()
          << callee << ": "
          << "expected result of interface (any) type but got " << resultType;
      }
    }
  }
  else if (callee == "unsafe.Add")
  {
    // Must have exactly 2 operands to be valid.
    if (this->getNumOperands() != 2)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected exactly 2 operands";
    }

    // Must return exactly ONE value.
    if (this->getNumResults() != 1)
    {
      return this->emitOpError() << callee << ": "
                                 << "expected 1 result";
    }

    // First operand MUST be an unsafe.Pointer.
    if (!isUnsafePointer(this->getOperand(0).getType()))
    {
      return this->emitOpError()
        << callee << ": "
        << "expected operand 0 to be an unsafe pointer (!go.ptr) type, but got "
        << this->getOperand(0).getType();
    }

    // Second operand MUST be an integer type.
    if (!isIntegerType(this->getOperand(1).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected operand 1 to be an integer type, but got "
                                 << this->getOperand(1).getType();
    }
  }
  else if (callee == "unsafe.Alignof")
  {
  }
  else if (callee == "unsafe.Offsetof")
  {
  }
  else if (callee == "unsafe.Sizeof")
  {
  }
  else if (callee == "unsafe.Slice")
  {
  }
  else if (callee == "unsafe.SliceData")
  {
  }
  else if (callee == "unsafe.String")
  {
  }
  else if (callee == "unsafe.StringData")
  {
  }
  else
  {
    return this->emitOpError() << "unknown built-in function \"" << callee << "\"";
  }
  return success();
}
} // namespace mlir::go
