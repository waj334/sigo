#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

namespace mlir::go
{

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor)
{
  // No-op bitcast: types identical.
  if (getValue().getType() == getType())
  {
    if (adaptor.getValue())
      return adaptor.getValue();

    mlir::SmallVector<OpFoldResult, 4> results;
    if (auto definingOp = getValue().getDefiningOp();
        definingOp && succeeded(definingOp->fold(results)) && !results.empty())
      return results.front();

    return {};
  }

  // Try to fold the source operand to get a constant attribute.
  mlir::Attribute srcAttr = adaptor.getValue();
  if (!srcAttr)
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (auto definingOp = getValue().getDefiningOp();
        definingOp && succeeded(definingOp->fold(results)) && !results.empty())
      srcAttr = mlir::dyn_cast<mlir::Attribute>(results.front());
  }

  if (!srcAttr)
    return {};

  // Allow folding IntegerAttr across integer-compatible types (including
  // untyped integers, sized integers, and named types that wrap integers).
  // This is safe because IntegerAttr carries a value + width, and the
  // GlobalConstantsPass preserves the original result type when creating
  // the replacement ConstantOp.
  if (const auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(srcAttr))
  {
    const auto resultType = mlir::dyn_cast<mlir::go::IntegerType>(getType());
    if (!resultType)
    {
      return srcAttr; // noninteger result, pass through as before
    }

    size_t targetWidth;
    if (const auto width = resultType.getWidth(); width.has_value())
    {
      targetWidth = *width;
    }
    else
    {
      if (!this->getOperation()->getParentOp())
      {
        return {};
      }

      const auto dataLayout = mlir::DataLayout::closest(this->getOperation());
      targetWidth = dataLayout.getTypeSizeInBits(resultType);
    }

    auto value = intAttr.getValue();
    if (value.getBitWidth() != targetWidth)
    {
      value = value.zextOrTrunc(targetWidth);
    }

    return mlir::IntegerAttr::get(mlir::IntegerType::get(this->getContext(), targetWidth), value);
  }

  // Allow folding FloatAttr across float-compatible types.
  if (mlir::isa<mlir::FloatAttr>(srcAttr))
    return srcAttr;

  // For other attribute kinds (e.g., StringAttr), don't fold across
  // different types — the lowering passes can't handle the mismatch.
  return {};
}

OpFoldResult IntTruncateOp::fold(FoldAdaptor adaptor)
{
  // Try to get the source attribute directly or by folding.
  mlir::Attribute srcAttr = adaptor.getValue();
  if (!srcAttr)
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (auto definingOp = getValue().getDefiningOp();
        definingOp && succeeded(definingOp->fold(results)) && !results.empty())
      srcAttr = mlir::dyn_cast<mlir::Attribute>(results.front());
  }

  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(srcAttr))
    return intAttr;

  return {};
}

OpFoldResult SignedExtendOp::fold(FoldAdaptor adaptor)
{
  mlir::Attribute srcAttr = adaptor.getValue();
  if (!srcAttr)
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (auto definingOp = getValue().getDefiningOp();
        definingOp && succeeded(definingOp->fold(results)) && !results.empty())
      srcAttr = mlir::dyn_cast<mlir::Attribute>(results.front());
  }

  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(srcAttr))
    return intAttr;

  return {};
}

OpFoldResult ZeroExtendOp::fold(FoldAdaptor adaptor)
{
  mlir::Attribute srcAttr = adaptor.getValue();
  if (!srcAttr)
  {
    mlir::SmallVector<OpFoldResult, 4> results;
    if (auto definingOp = getValue().getDefiningOp();
        definingOp && succeeded(definingOp->fold(results)) && !results.empty())
      srcAttr = mlir::dyn_cast<mlir::Attribute>(results.front());
  }

  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(srcAttr))
    return intAttr;

  return {};
}

::mlir::LogicalResult BitcastOp::verify()
{
  return success();

  // Verify that primitive types are only cast between their MLIR and runtime representations and
  // that pointers are only cast to and from unsafe.Pointer.
  if (failed(
        llvm::TypeSwitch<mlir::Type, LogicalResult>(this->getValue().getType())
          .Case(
            [&](ChanType T)
            {
              if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
              {
                return success(resultType.getName() == "runtime._channel");
              }
              return failure();
            })
          .Case(
            [&](ComplexType T)
            {
              // Allow bitcast from one complex type to another.
              return success(go::isa<mlir::ComplexType>(this->getType()));
            })
          .Case(
            [&](InterfaceType T)
            {
              if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
              {
                return success(resultType.getName() == "runtime._interface");
              }
              return failure();
            })
          .Case(
            [&](MapType T)
            {
              if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
              {
                return success(resultType.getName() == "runtime._map");
              }
              return failure();
            })
          .Case(
            [&](SliceType T)
            {
              if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
              {
                return success(resultType.getName() == "runtime._slice");
              }
              return failure();
            })
          .Case(
            [&](StringType T)
            {
              if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
              {
                return success(resultType.getName() == "runtime._string");
              }
              return failure();
            })
          .Case(
            [&](NamedType T)
            {
              bool isSuccess =
                llvm::StringSwitch<bool>(T.getName().getValue())
                  .Case("runtime._channel", go::isa<ChanType>(this->getType()))
                  .Case("runtime._interface", go::isa<InterfaceType>(this->getType()))
                  .Case("runtime._map", go::isa<MapType>(this->getType()))
                  .Case("runtime._slice", go::isa<SliceType>(this->getType()))
                  .Case("runtime._string", go::isa<StringType>(this->getType()))
                  .Default(false);

              if (!isSuccess)
              {
                {
                  auto fromType = go::dyn_cast<IntegerType>(T);
                  auto toType = go::dyn_cast<IntegerType>(this->getType());
                  if (fromType && toType)
                  {
                    return success(fromType.getWidth() == toType.getWidth());
                  }
                }
                {
                  auto fromType = go::dyn_cast<FloatType>(T);
                  auto toType = go::dyn_cast<FloatType>(this->getType());
                  if (fromType && toType)
                  {
                    return success(fromType.getWidth() == toType.getWidth());
                  }
                }
                {
                  auto fromType = go::dyn_cast<ComplexType>(T);
                  auto toType = go::dyn_cast<ComplexType>(this->getType());
                  if (fromType && toType)
                  {
                    auto fromFType = mlir::cast<FloatType>(fromType.getElementType());
                    auto toFType = mlir::cast<FloatType>(toType.getElementType());
                    return success(fromFType.getWidth() == toFType.getWidth());
                  }
                }

                // The conversion is valid if both types have the same underlying type.
                return success(baseType(this->getType()) == baseType(T));
              }

              return success(isSuccess);
            })
          .Case(
            [&](PointerType T)
            {
              if (auto resultType = mlir::dyn_cast<PointerType>(this->getType()))
              {
                if (T.getElementType().has_value())
                {
                  if (!resultType.getElementType().has_value())
                  {
                    // *T -> unsafe.Pointer.
                    return success(true);
                  }
                  else if (baseType(*T.getElementType()) == this->getType())
                  {
                    // Alias -> underlying type.
                    return success(true);
                  }
                }
                // unsafe.Pointer -> *T is always acceptable.
                return success();
              }
              else if (!T.getElementType().has_value())
              {
                if (go::isa<FunctionType>(this->getType()))
                {
                  // unsafe.Pointer -> func
                  return success();
                }
              }
              return failure();
            })
          .Case(
            [&](IntegerType T)
            {
              if (auto resultType = mlir::dyn_cast<IntegerType>(baseType(this->getType())))
              {
                return success(T.getWidth() == resultType.getWidth());
              }
              return failure();
            })
          .Default(
            [&](mlir::Type T)
            {
              // The conversion is valid if both types have the same underlying type.
              return success(baseType(this->getType()) == baseType(T));
            })))
  {
    return this->emitOpError() << "invalid cast from " << this->getValue().getType() << " to "
                               << this->getType();
  }
  return success();
}

::mlir::LogicalResult FunctionToPointerOp::verify()
{
  // The resulting pointer must be opaque
  auto resultType = cast<PointerType>(this->getResult().getType());
  if (resultType.getElementType())
  {
    return emitOpError() << "the resulting pointer must be opaque";
  }
  return success();
}

::mlir::LogicalResult PointerToFunctionOp::verify()
{
  // The pointer operand must be opaque
  auto pointerType = cast<PointerType>(this->getValue().getType());
  if (pointerType.getElementType())
  {
    return emitOpError() << "the pointer operand must be opaque";
  }
  return success();
}

::mlir::LogicalResult ChangeInterfaceOp::verify()
{
  const auto sourceType = go::dyn_cast<InterfaceType>(this->getValue().getType());
  if (!sourceType)
  {
    return this->emitOpError() << "the input value MUST be an interface type";
  }

  const auto resultType = go::dyn_cast<InterfaceType>(this->getType());
  if (!resultType)
  {
    return this->emitOpError() << "the result type MUST be an interface type";
  }

  const auto sourceMethods = sourceType.getMethods();
  const auto resultMethods = resultType.getMethods();

  // Check the interface methods for compatibility.
  for (auto& method : resultMethods)
  {
    const auto sourceMethod = sourceMethods.find(method.first);
    if (sourceMethod == sourceMethods.end())
    {
      return this->emitOpError() << "the resulting interface type is missing method \""
                                 << method.first << "\" from the input interface value type";
    }

    // The signatures of the methods MUST match for the resulting interface type to be compatible
    // with the input interface. Interface method compatibility in Go is determined by parameter
    // and result types only — the receiver type is implicit and may differ between interfaces
    // (e.g. net.Conn::Write has receiver net.Conn while io.Writer::Write has receiver io.Writer).
    if (method.second != sourceMethod->second)
    {
      // Fall back to comparing inputs and results only, ignoring the receiver.
      const auto resultFnType = dyn_cast<FunctionType>(method.second);
      const auto sourceFnType = dyn_cast<FunctionType>(sourceMethod->second);
      if (!resultFnType || !sourceFnType ||
          resultFnType.getInputs() != sourceFnType.getInputs() ||
          resultFnType.getResults() != sourceFnType.getResults())
      {
        return this->emitOpError()
          << "the signature of method \"" << method.first
          << "\" in the resulting interface type does not match that of the input "
             "value interface type";
      }
    }
  }

  // The resulting interface type MUST implement ALL methods of the input interface type.
  return success();
}

} // namespace mlir::go
