#pragma once

#include <utility>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/TypeID.h>

#include "Go/IR/Types/InterfaceDetail.h"

namespace mlir::go
{

class InterfaceType
  : public ::mlir::Type::TypeBase<
      InterfaceType,
      ::mlir::Type,
      detail::InterfaceTypeStorage,
      ::mlir::DataLayoutTypeInterface::Trait,
      TypeTrait::IsMutable>
{
public:
  using Base::Base;

  static InterfaceType get(
    ::mlir::MLIRContext* context,
    const detail::InterfaceTypeStorage::FunctionMap& methods);

  static InterfaceType getNamed(::mlir::MLIRContext* context, const std::string& name);

  static constexpr ::llvm::StringLiteral name = "go.interface";

  static constexpr ::llvm::StringLiteral getMnemonic() { return { "interface" }; }

  static ::mlir::Type parse(::mlir::AsmParser& p);

  void print(::mlir::AsmPrinter& p) const;

  mlir::LogicalResult setMethods(const detail::InterfaceTypeStorage::FunctionMap& methods);

  [[nodiscard]] detail::InterfaceTypeStorage::FunctionMap getMethods() const;

  [[nodiscard]] std::string getName() const;

  ::llvm::TypeSize getTypeSize(
    const ::mlir::DataLayout& dataLayout,
    ::mlir::DataLayoutEntryListRef params) const;
  ::llvm::TypeSize getTypeSizeInBits(
    const ::mlir::DataLayout& dataLayout,
    ::mlir::DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(
    const ::mlir::DataLayout& dataLayout,
    ::mlir::DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(
    const ::mlir::DataLayout& dataLayout,
    ::mlir::DataLayoutEntryListRef params) const;
};

} // namespace mlir::go

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::go::InterfaceType)
