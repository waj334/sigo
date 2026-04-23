#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

namespace mlir::go
{

struct PointerTypeStorage : public ::mlir::TypeStorage
{
  // Key: element type + optional DistinctAttr for deferred pointers.
  // Normal pointers are keyed by element type only (id is null).
  // Deferred pointers are keyed by DistinctAttr only.
  using KeyTy = std::tuple<std::optional<Type>, mlir::StringRef>;

  // Normal pointer (with element type, possibly nullopt for unsafe.Pointer).
  explicit PointerTypeStorage(std::optional<Type> elementType)
    : m_elementType(std::move(elementType))
    , m_complete(true)
  {
  }

  // Deferred pointer (keyed by DistinctAttr, element type set later).
  explicit PointerTypeStorage(mlir::StringRef id)
    : m_id(std::move(id))
    , m_complete(false)
  {
  }

  bool operator==(const KeyTy& key) const
  {
    const auto& id = std::get<1>(key);
    if (!id.empty())
    {
      return m_id == id;
    }
    return m_elementType == std::get<0>(key) && m_id.empty();
  }

  static llvm::hash_code hashKey(const KeyTy& key)
  {
    const auto& id = std::get<1>(key);
    if (!id.empty())
    {
      return llvm::hash_combine(id);
    }
    return llvm::hash_combine(std::get<0>(key));
  }

  static PointerTypeStorage* construct(TypeStorageAllocator& allocator, KeyTy&& key)
  {
    auto id = std::get<1>(key);
    if (!id.empty())
    {
      // Copy the string into the allocator so the StringRef outlives the caller's buffer.
      id = allocator.copyInto(id);
      return new (allocator.allocate<PointerTypeStorage>()) PointerTypeStorage(id);
    }
    return new (allocator.allocate<PointerTypeStorage>())
      PointerTypeStorage(std::get<0>(std::move(key)));
  }

  LogicalResult mutate(TypeStorageAllocator&, std::optional<Type> elementType)
  {
    if (m_complete || m_id.empty())
    {
      return failure();
    }
    m_elementType = std::move(elementType);
    m_complete = true;
    return success();
  }

  std::optional<Type> m_elementType;
  mlir::StringRef m_id;       // Non-null only for deferred pointers.
  bool m_complete = false;
};

} // namespace mlir::go
