#pragma once

#include "Go/IR/GoTypes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/TypeID.h>

#include <utility>

namespace mlir::go {

namespace detail {
struct InterfaceTypeStorage : public ::mlir::TypeStorage {
  using FunctionMap = std::map<std::string, FunctionType>;

  struct Key {
    explicit Key(const std::string &name) : m_name(name) {}

    explicit Key(FunctionMap methods) : m_methods(std::move(methods)) {}

    [[nodiscard]] ::llvm::hash_code hashKey() const {
      llvm::hash_code result{};
      if (!this->m_name.empty()) {
        result = llvm::hash_value(this->m_name);
      } else {
        // Hash each field
        for (auto field : this->m_methods) {
          result = ::llvm::hash_combine(result, field);
        }
      }
      return result;
    }

    std::string m_name;
    FunctionMap m_methods;
  };

  using KeyTy = Key;

  explicit InterfaceTypeStorage(const std::string &name);

  explicit InterfaceTypeStorage(const FunctionMap &methods);

  bool operator==(const KeyTy &key) const;

  [[nodiscard]] KeyTy getAsKey() const;

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] FunctionMap getMethods() const;

  static ::llvm::hash_code hashKey(const KeyTy &key);

  static InterfaceTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key);

  LogicalResult mutate(TypeStorageAllocator &allocator, const FunctionMap &methods);

  std::string m_name;
  FunctionMap m_methods;
  bool m_isSet = false;
};
} // namespace detail
class InterfaceType : public ::mlir::Type::TypeBase<InterfaceType, ::mlir::Type, detail::InterfaceTypeStorage,
                                                    ::mlir::DataLayoutTypeInterface::Trait, TypeTrait::IsMutable> {
public:
  using Base::Base;

  static InterfaceType get(::mlir::MLIRContext *context, const detail::InterfaceTypeStorage::FunctionMap &methods);

  static InterfaceType getNamed(::mlir::MLIRContext *context, const std::string &name);

  static constexpr ::llvm::StringLiteral name = "go.interface";

  static constexpr ::llvm::StringLiteral getMnemonic() { return {"interface"}; }

  static ::mlir::Type parse(::mlir::AsmParser &p);

  void print(::mlir::AsmPrinter &p) const;

  mlir::LogicalResult setMethods(const detail::InterfaceTypeStorage::FunctionMap &methods);

  [[nodiscard]] detail::InterfaceTypeStorage::FunctionMap getMethods() const;

  [[nodiscard]] std::string getName() const;

  ::llvm::TypeSize getTypeSize(const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  ::llvm::TypeSize getTypeSizeInBits(const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;
};

} // namespace mlir::go

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::go::InterfaceType)
