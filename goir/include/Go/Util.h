#pragma once

#include <Go/IR/GoTypes.h>
#include <string>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

namespace mlir::go {

enum class GoTypeId : uint8_t {
  Invalid = 0,
  Bool,
  Int,
  Int8,
  Int16,
  Int32,
  Int64,
  Uint,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Uintptr,
  Float32,
  Float64,
  Complex64,
  Complex128,
  Array,
  Chan,
  Func,
  Interface,
  Map,
  Pointer,
  Slice,
  String,
  Struct,
  UnsafePointer
};

GoTypeId GetGoTypeId(const mlir::Type& type);

//std::string typeInfoSymbol(const mlir::Type &type, const std::string &postfix = "_");

//uint64_t getTypeId(const mlir::Type &type);

std::string typeStr(const mlir::Type &T);

llvm::hash_code computeMethodHash(const StringRef name, const FunctionType func, bool isInterface);

inline uint64_t alignTo(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

} // namespace mlir::go
