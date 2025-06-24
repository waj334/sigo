#include "Go/IR/GoInterfaces.h"

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoTypes.h"

namespace mlir::go
{

bool isCompatibleUntypedKind(mlir::go::UntypedBasicKind kind, mlir::Type targetType)
{
  using namespace mlir::go;

  switch (kind)
  {
    case mlir::go::UntypedBasicKind::Boolean:
      return mlir::isa<BooleanType>(targetType);
    case mlir::go::UntypedBasicKind::Integer:
      return mlir::isa<IntegerType>(targetType);
    case mlir::go::UntypedBasicKind::Rune:
    {
      if (!mlir::isa<IntegerType>(targetType))
      {
        return false;
      }
      const auto intT = mlir::dyn_cast<IntegerType>(targetType);
      return intT.getWidth() && *intT.getWidth() == 32;
    }
    case mlir::go::UntypedBasicKind::Float:
      return mlir::isa<FloatType>(targetType);
    case mlir::go::UntypedBasicKind::Complex:
      return mlir::isa<ComplexType>(targetType);
    case mlir::go::UntypedBasicKind::String:
      return mlir::isa<StringType>(targetType);
    case mlir::go::UntypedBasicKind::Nil:
      return mlir::isa<PointerType, InterfaceType, ChanType, MapType>(targetType);
  }

  return false;
}

mlir::LogicalResult verifyHasCompatibleOperandsTypesTrait(mlir::Operation* op)
{
  if (op->getOperands().empty())
  {
    return mlir::success();
  }

  mlir::Type lastOperandType;
  for (const auto& operand : op->getOperands())
  {
    const auto& operandType = operand.getType();
    if (mlir::isa<mlir::go::UntypedType>(operandType))
    {
      // TODO: Enforce that the default type is the same class of type.
      continue;
    }

    if (lastOperandType && operandType != lastOperandType)
    {
      return op->emitError("operand type mismatch: expected ")
        << lastOperandType << ", got " << operandType;
    }

    lastOperandType = operandType;
  }

  return mlir::success();
}

mlir::LogicalResult verifyHasCompatibleOperandsAndResultTypesTrait(mlir::Operation* op)
{
  if (op->getOperands().empty())
  {
    return mlir::success();
  }

  mlir::Type lastOperandType;
  for (const auto& operand : op->getOperands())
  {
    const auto& operandType = operand.getType();
    if (mlir::isa<mlir::go::UntypedType>(operandType))
    {
      // TODO: Enforce that the default type is the same class of type.
      continue;
    }

    if (lastOperandType && operandType != lastOperandType)
    {
      return op->emitError("operand type mismatch: expected ")
        << lastOperandType << ", got " << operandType;
    }

    lastOperandType = operandType;
  }

  if (lastOperandType)
  {
    for (const auto& resultType : op->getResultTypes())
    {
      if (mlir::isa<mlir::go::UntypedType>(resultType))
      {
        // TODO: Enforce that the default type is the same class of type.
        continue;
      }

      if (resultType != lastOperandType)
      {
        return op->emitError("result type mismatch: expected ")
          << lastOperandType << ", got " << resultType;
      }
    }
  }

  return mlir::success();
}

mlir::SmallVector<mlir::Type> defaultResolveOperandTypes(mlir::Operation* op)
{
  auto result = mlir::SmallVector<mlir::Type>(op->getOperandTypes());
  for (size_t i = 0; i < result.size(); ++i)
  {
    if (const auto untypedType = mlir::dyn_cast<mlir::go::UntypedType>(result[i]))
    {
      // Use the default type.
      result[i] = untypedType.getDefaultType();
    }
  }
  return result;
}

mlir::SmallVector<mlir::Type> defaultResolveResultTypes(mlir::Operation* op)
{
  auto result = mlir::SmallVector<mlir::Type>(op->getResultTypes());
  for (size_t i = 0; i < result.size(); ++i)
  {
    if (const auto untypedType = mlir::dyn_cast<mlir::go::UntypedType>(result[i]))
    {
      // Use the default type.
      result[i] = untypedType.getDefaultType();
    }
  }
  return result;
}

} // namespace mlir::go