#ifndef GO_GOTYPES_H
#define GO_GOTYPES_H

#include <optional>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/TypeID.h>

#include "Go/Interfaces/ReflectTypeInterface.h"

#include "Go/IR/GoDialect.h"
#include "Go/IR/Types/Interface.h"
// #include "Go/IR/Types/Struct.h"

#include "llvm/Support/TypeSize.h"

#define GET_TYPEDEF_CLASSES
#include "Go/IR/GoTypes.h.inc"

namespace mlir::go {

inline Type underlyingType(Type type) {
  if (const auto named = ::llvm::dyn_cast<NamedType>(type); named) {
    return named.getUnderlying();
  }
  return type;
}

inline Type baseType(Type type) {
  if (const auto named = ::llvm::dyn_cast<NamedType>(type); named) {
    return baseType(named.getUnderlying());
  }
  return type;
}

template <typename T> T cast(const Type type) { return ::llvm::cast<T>(underlyingType(type)); }
template <typename T> T dyn_cast(const Type type) { return ::llvm::dyn_cast<T>(underlyingType(type)); }
template <typename T> bool isa(const Type type) { return ::llvm::isa<T>(underlyingType(type)); }

inline llvm::TypeSize getDefaultTypeSize(Type type, const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) {
  if (mlir::isa<FunctionType>(type)) {
    return mlir::detail::getDefaultTypeSize(PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultTypeSize(type, dataLayout, params);
}

inline llvm::TypeSize getDefaultTypeSizeInBits(Type type, const DataLayout &dataLayout, DataLayoutEntryListRef params) {
  if (mlir::isa<FunctionType>(type)) {
    return mlir::detail::getDefaultTypeSizeInBits(PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout, params);
}

inline uint64_t getDefaultABIAlignment(Type type, const DataLayout &dataLayout,
                                ArrayRef<DataLayoutEntryInterface> params) {
  if (mlir::isa<FunctionType>(type)) {
    return mlir::detail::getDefaultABIAlignment(PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
}

inline uint64_t getDefaultPreferredAlignment(Type type, const DataLayout &dataLayout,
                                ArrayRef<DataLayoutEntryInterface> params) {
  if (mlir::isa<FunctionType>(type)) {
    return mlir::detail::getDefaultPreferredAlignment(PointerType::get(type.getContext(), {}), dataLayout, params);
  }
  return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout, params);
}

} // namespace mlir::go

#endif // GO_GOTYPES_H