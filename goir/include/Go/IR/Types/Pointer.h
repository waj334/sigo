#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <Go/IR/Types/PointerDetail.h>

namespace mlir::go
{

class PointerType
  : public ::mlir::Type::TypeBase<
      PointerType,
      ::mlir::Type,
      PointerTypeStorage,
      ::mlir::DataLayoutTypeInterface::Trait,
      TypeTrait::IsMutable>
{
public:
  using Base::Base;

  static constexpr ::llvm::StringLiteral name = "go.ptr";

  /// Create a normal pointer with a known element type (or nullopt for unsafe.Pointer).
  static PointerType get(MLIRContext* context, std::optional<Type> elementType = {});

  /// Create a deferred pointer keyed by a unique DistinctAttr.
  /// The element type must be set later via setElementType().
  static PointerType getDeferred(MLIRContext* context, mlir::StringRef id);

  static constexpr ::llvm::StringLiteral getMnemonic() { return { "ptr" }; }

  [[nodiscard]] std::optional<Type> getElementType() const;

  /// Set the element type of a deferred pointer. Fails if the pointer is
  /// already complete or was not created via getDeferred().
  LogicalResult setElementType(std::optional<Type> elementType);

  /// Returns true if this pointer was created via getDeferred() and has not
  /// yet had its element type set.
  [[nodiscard]] bool isDeferred() const;

  static ::mlir::Type parse(::mlir::AsmParser& p);
  void print(::mlir::AsmPrinter& p) const;

  /// DataLayoutTypeInterface methods.
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

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::go::PointerType)
