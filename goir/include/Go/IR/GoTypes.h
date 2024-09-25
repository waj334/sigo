#ifndef GO_GOTYPES_H
#define GO_GOTYPES_H

#include <optional>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/Types/Interface.h"
#include "Go/IR/Types/Struct.h"
#include "llvm/Support/TypeSize.h"

#define GET_TYPEDEF_CLASSES
#include <llvm/ADT/TypeSwitch.h>

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

inline bool isIntegerType(const Type type)
{
  return go::isa<IntegerType>(type);
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
  return isIntegerType(type) || go::isa<FloatType>(type) || go::isa<StringType>(type);
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

inline llvm::TypeSize
getDefaultTypeSize(Type type, const DataLayout& dataLayout, DataLayoutEntryListRef params)
{
  if (mlir::isa<FunctionType>(type))
  {
    return mlir::detail::getDefaultTypeSize(
      PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultTypeSize(type, dataLayout, params);
}

inline llvm::TypeSize
getDefaultTypeSizeInBits(Type type, const DataLayout& dataLayout, DataLayoutEntryListRef params)
{
  if (mlir::isa<FunctionType>(type))
  {
    return mlir::detail::getDefaultTypeSizeInBits(
      PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout, params);
}

inline uint64_t getDefaultABIAlignment(
  Type type,
  const DataLayout& dataLayout,
  ArrayRef<DataLayoutEntryInterface> params)
{
  if (mlir::isa<FunctionType>(type))
  {
    return mlir::detail::getDefaultABIAlignment(
      PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
}

inline uint64_t getDefaultPreferredAlignment(
  Type type,
  const DataLayout& dataLayout,
  ArrayRef<DataLayoutEntryInterface> params)
{
  if (mlir::isa<FunctionType>(type))
  {
    return mlir::detail::getDefaultPreferredAlignment(
      PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout, params);
}

} // namespace mlir::go

#endif // GO_GOTYPES_H
