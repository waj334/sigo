#ifndef GO_GOTYPES_H
#define GO_GOTYPES_H

#include <optional>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/TypeSize.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Go/IR/GoAttrs.h"
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoEnums.h"
#include "Go/IR/GoInterfaces.h"
#include "Go/IR/Types/Interface.h"
#include "Go/IR/Types/Struct.h"

#define GET_TYPEDEF_CLASSES
#include "Go/IR/GoTypes.h.inc"

namespace mlir::go
{

inline Type underlyingType(Type type)
{
  if (const auto named = ::mlir::dyn_cast<NamedType>(type); named)
  {
    return named.getUnderlying();
  }
  return type;
}

inline Type baseType(Type type)
{
  if (const auto named = ::mlir::dyn_cast<NamedType>(type); named)
  {
    return baseType(named.getUnderlying());
  }
  return type;
}

template<typename T>
T cast(const Type type)
{
  return ::mlir::cast<T>(baseType(type));
}

template<typename T>
T dyn_cast(const Type type)
{
  return ::mlir::dyn_cast<T>(baseType(type));
}

template<typename T>
bool isa(const Type type)
{
  return ::mlir::isa<T>(baseType(type));
}

template<typename DesiredT, typename ActualT>
bool isCompatibleType(const ActualT actual)
{
  if (mlir::go::isa<DesiredT>(actual))
  {
    return true;
  }

  if (const auto untyped = mlir::go::dyn_cast<mlir::go::UntypedType>(actual))
  {
    using mlir::go::UntypedBasicKind;

    const auto basic = untyped.getBasicKind().getValue();
    if constexpr (std::is_same_v<DesiredT, mlir::go::BooleanType>)
      return basic == UntypedBasicKind::Boolean;
    else if constexpr (std::is_same_v<DesiredT, mlir::FloatType>)
      return basic == UntypedBasicKind::Float;
    else if constexpr (std::is_same_v<DesiredT, mlir::go::IntegerType>)
      return basic == UntypedBasicKind::Integer || basic == UntypedBasicKind::Rune;
    else if constexpr (std::is_same_v<DesiredT, mlir::ComplexType>)
      return basic == UntypedBasicKind::Complex;
    else if constexpr (std::is_same_v<DesiredT, mlir::go::StringType>)
      return basic == UntypedBasicKind::String;
    else if constexpr (
      std::is_same_v<DesiredT, mlir::go::InterfaceType> ||
      std::is_same_v<DesiredT, mlir::go::PointerType> ||
      std::is_same_v<DesiredT, mlir::go::MapType> || std::is_same_v<DesiredT, mlir::go::ChanType> ||
      std::is_same_v<DesiredT, mlir::go::SliceType> ||
      std::is_same_v<DesiredT, mlir::go::FunctionType>)
      return basic == UntypedBasicKind::Nil;
    else
      return false;
  }

  return false;
}

inline bool isCompatibleType(const mlir::Type actual, const mlir::Type expected)
{
  if (actual == expected)
  {
    return true;
  }

  if (const auto untypedType = mlir::go::dyn_cast<mlir::go::UntypedType>(actual))
  {
    const auto basicKind = untypedType.getBasicKind().getValue();
    return mlir::TypeSwitch<mlir::Type, bool>(expected)
      .Case(
        [&](mlir::go::IntegerType)
        {
          return basicKind == mlir::go::UntypedBasicKind::Integer ||
            basicKind == mlir::go::UntypedBasicKind::Rune;
        })
      .Case([&](mlir::go::BooleanType) { return basicKind == mlir::go::UntypedBasicKind::Boolean; })
      .Case([&](mlir::ComplexType) { return basicKind == mlir::go::UntypedBasicKind::Complex; })
      .Case([&](mlir::FloatType) { return basicKind == mlir::go::UntypedBasicKind::Float; })
      .Case([&](mlir::go::StringType) { return basicKind == mlir::go::UntypedBasicKind::String; })
      .Case([&](mlir::go::IntegerType) { return basicKind == mlir::go::UntypedBasicKind::Integer; })
      .Case<
        mlir::go::InterfaceType,
        mlir::go::PointerType,
        mlir::go::MapType,
        mlir::go::ChanType,
        mlir::go::SliceType,
        mlir::go::FunctionType>([&](auto) { return basicKind == mlir::go::UntypedBasicKind::Nil; })
      .Case([&](mlir::go::NamedType T) { return isCompatibleType(actual, T.getUnderlying()); })
      .Default([&](mlir::Type) { return false; });
  }

  return false;
}

inline bool isIntegerType(const Type type)
{
  return isCompatibleType<mlir::go::IntegerType>(type);
}

inline bool isUnsigned(const Type type)
{
  if (const auto intType = go::dyn_cast<IntegerType>(type); intType)
  {
    return intType.isUnsigned();
  }
  return false;
}

inline bool isOrderedType(const Type type)
{
  return isCompatibleType<mlir::go::IntegerType>(type) || isCompatibleType<mlir::FloatType>(type) ||
    isCompatibleType<mlir::go::StringType>(type);
}

inline bool isAnyType(const Type type)
{
  if (!go::isa<::mlir::go::InterfaceType>(type))
  {
    return false;
  }

  const auto interfaceType = ::mlir::dyn_cast<::mlir::go::InterfaceType>(type);

  // Any type has no methods.
  return interfaceType.getMethods().size() == 0;
}

inline bool isUnsafePointer(Type type)
{
  if (!go::isa<::mlir::go::PointerType>(type))
  {
    return false;
  }
  const auto pointerType = go::dyn_cast<::mlir::go::PointerType>(type);
  return pointerType.getElementType() == std::nullopt;
}

} // namespace mlir::go

#endif // GO_GOTYPES_H
