#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

constexpr std::string_view runtimeFuncTypeName = "runtime._func";

namespace mlir::go
{
::mlir::LogicalResult DeferOp::verify()
{
  FunctionType Fn;
  size_t offset = 0;

  auto result =
    llvm::TypeSwitch<mlir::Type, LogicalResult>(baseType(this->getCallee().getType()))
      .Case<FunctionType>(
        [&](auto T) -> LogicalResult
        {
          if (this->getMethodNameAttr())
          {
            return this->emitOpError()
              << "a method can only be specified if the callee is an interface";
          }

          Fn = T;
          return success();
        })
      .Case<InterfaceType>(
        [&](InterfaceType T) -> LogicalResult
        {
          if (!this->getMethodNameAttr())
          {
            return this->emitOpError()
              << "a method must be specified if the callee is an interface";
          }

          auto methods = T.getMethods();
          if (auto it = methods.find(this->getMethodNameAttr().str()); it != methods.end())
          {
            Fn = mlir::cast<FunctionType>(it->second);

            // Skip the receiver value.
            offset = 1;

            return success();
          }
          return this->emitOpError() << "interface has no method " << this->getMethodNameAttr();
        })
      .Case(
        [&](PointerType T) -> LogicalResult
        {
          if (auto elementType = T.getElementType())
          {
            // Must be a pointer to a function.
            if (mlir::go::isa<FunctionType>(*elementType))
            {
              Fn = mlir::go::dyn_cast<mlir::go::FunctionType>(*elementType);
              return success();
            }
          }
          return this->emitOpError() << "pointer must be to a function. got " << T;
        })
      .Case(
        [&](GoStructType) -> LogicalResult
        {
          // The struct must be the "func" struct type.
          auto namedType = mlir::dyn_cast<NamedType>(this->getCallee().getType());
          if (namedType && namedType.getName() == "runtime.func")
          {
            return success();
          }
          return this->emitOpError()
            << "expected \"runtime.func\" struct type, but got " << namedType;
        })
      .Default([&](auto T) { return this->emitOpError() << "unsupported callee type " << T; });

  if (failed(result))
  {
    return result;
  }

  if (Fn)
  {
    // Assert args actually match the function operand's signature
    if (this->getCalleeOperands().size() != Fn.getNumInputs())
    {
      return this->emitOpError() << "number of operands does not match function signature";
    }

    // Check each type to ensure that they match the function signature
    for (size_t i = offset; i < Fn.getNumInputs(); i++)
    {
      if (Fn.getInput(i) != this->getCalleeOperands()[i].getType())
      {
        return this->emitOpError() << "operand type " << i << " does not match signature";
      }
    }
  }
  return success();
}

::mlir::LogicalResult GoOp::verify()
{
  FunctionType Fn;
  size_t offset = 0;

  auto result =
    llvm::TypeSwitch<mlir::Type, LogicalResult>(baseType(this->getCallee().getType()))
      .Case(
        [&](FunctionType T) -> LogicalResult
        {
          if (this->getMethodNameAttr())
          {
            return this->emitOpError()
              << "a method can only be specified if the callee is an interface";
          }

          Fn = T;
          return success();
        })
      .Case(
        [&](InterfaceType T) -> LogicalResult
        {
          if (!this->getMethodNameAttr())
          {
            return this->emitOpError()
              << "a method must be specified if the callee is an interface";
          }

          auto methods = T.getMethods();
          if (auto it = methods.find(this->getMethodNameAttr().str()); it != methods.end())
          {
            Fn = mlir::cast<FunctionType>(it->second);

            // Skip the receiver value.
            offset = 1;

            return success();
          }
          return this->emitOpError() << "interface has no method " << this->getMethodNameAttr();
        })
      .Case(
        [&](PointerType T) -> LogicalResult
        {
          if (auto elementType = T.getElementType())
          {
            // Must be a pointer to a function.
            if (mlir::go::isa<FunctionType>(*elementType))
            {
              Fn = mlir::go::dyn_cast<mlir::go::FunctionType>(*elementType);
              return success();
            }
          }
          return this->emitOpError() << "pointer must be to a function. got " << T;
        })
      .Case(
        [&](GoStructType) -> LogicalResult
        {
          // The struct must be the "func" struct type.
          auto namedType = mlir::dyn_cast<NamedType>(this->getCallee().getType());
          if (namedType && namedType.getName() == runtimeFuncTypeName)
          {
            return success();
          }
          return this->emitOpError() << "expected \"" << runtimeFuncTypeName << "\" struct type, but got " << namedType;
        })
      .Default([&](auto T) { return this->emitOpError() << "unsupported callee type " << T; });

  if (failed(result))
  {
    return result;
  }

  if (Fn)
  {
    // Assert args actually match the function operand's signature
    if (this->getCalleeOperands().size() != Fn.getNumInputs())
    {
      return this->emitOpError() << "number of operands does not match function signature";
    }

    // Check each type to ensure that they match the function signature
    for (size_t i = offset; i < Fn.getNumInputs(); i++)
    {
      if (Fn.getInput(i) != this->getCalleeOperands()[i].getType())
      {
        return this->emitOpError() << "operand type " << i << " does not match signature";
      }
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
    if (!go::isa<SliceType>(this->getOperand(0).getType()))
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
      !go::isa<ArrayType>(inputType) && !go::isa<ChanType>(inputType) &&
      !go::isa<SliceType>(inputType))
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
    if (!go::isa<MapType>(inputType) && !go::isa<SliceType>(inputType))
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
    if (!go::isa<ChanType>(inputType))
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
    if (!go::isa<FloatType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected floating-point operand type for operand 0 but got "
                                 << this->getOperand(0).getType();
    }

    if (!go::isa<FloatType>(this->getOperand(1).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected floating-point operand type for operand 1 but got "
                                 << this->getOperand(1).getType();
    }

    // The result MUST be a complex number type.
    if (!go::isa<ComplexType>(this->getResult(0).getType()))
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
    if (!go::isa<SliceType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected slice but got " << this->getOperand(0).getType()
                                 << "for operand 0";
    }

    const auto inputSliceType = go::cast<SliceType>(this->getOperand(1).getType());

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

    // Input value MUST be a map.
    const auto inputType = this->getOperand(0).getType();
    if (!go::isa<MapType>(inputType))
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
    if (!go::isa<ComplexType>(this->getOperand(0).getType()))
    {
      return this->emitOpError() << callee << ": "
                                 << "expected complex operand type but got "
                                 << this->getOperand(0).getType();
    }

    // The result MUST be a float type.
    if (!go::isa<FloatType>(this->getResult(0).getType()))
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
      !go::isa<ArrayType>(inputType) && !go::isa<ChanType>(inputType) &&
      !go::isa<MapType>(inputType) && !go::isa<SliceType>(inputType) &&
      !go::isa<StringType>(inputType))
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
    if (!go::isa<PointerType>(resultType))
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
