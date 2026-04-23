#include "Go/IR/Types/Pointer.h"

#include <mlir/IR/OpImplementation.h>

#include "Go/IR/GoTypes.h"

namespace mlir::go
{

PointerType PointerType::get(MLIRContext* context, std::optional<Type> elementType)
{
  // When the element type is a NamedType, route through getDeferred() so that
  // all pointers to the same named type share a single type identity regardless
  // of whether the caller used get() or getDeferred().
  if (elementType)
  {
    if (auto namedType = mlir::dyn_cast<NamedType>(*elementType))
    {
      auto name = namedType.getName().getValue();
      auto ptrType = getDeferred(context, name);
      if (!ptrType.getElementType())
      {
        ptrType.setElementType(elementType);
      }
      return ptrType;
    }
  }
  return Base::get(context, elementType, mlir::StringRef{});
}

PointerType PointerType::getDeferred(MLIRContext* context, mlir::StringRef id)
{
  assert(!id.empty() && "StringRef must not be empty for deferred pointer");
  return Base::get(context, std::nullopt, id);
}

std::optional<Type> PointerType::getElementType() const
{
  return this->getImpl()->m_elementType;
}

LogicalResult PointerType::setElementType(std::optional<Type> elementType)
{
  return this->mutate(elementType);
}

bool PointerType::isDeferred() const
{
  return !this->getImpl()->m_id.empty();
}

::mlir::Type PointerType::parse(::mlir::AsmParser& p)
{
  if (failed(p.parseOptionalLess()))
  {
    // No `<`: unsafe.Pointer
    return PointerType::get(p.getContext(), std::nullopt);
  }

  // Try to parse a DistinctAttr (deferred/recursive pointer).
  // DistinctAttr starts with the keyword `distinct`, which cannot begin a type,
  // so parseOptionalAttribute safely backtracks if not present.
  std::string idStr;
  if (const auto parseResult = p.parseKeywordOrString(&idStr); succeeded(parseResult))
  {
    auto ptrType = PointerType::getDeferred(p.getContext(), idStr);
    auto cyclicParse = p.tryStartCyclicParse(ptrType);
    if (failed(cyclicParse))
    {
      // Back-reference: type already being parsed upstream — close and return.
      if (p.parseGreater())
      {
        p.emitError(p.getCurrentLocation(), "expected `>`");
        return {};
      }
      return ptrType;
    }

    // First occurrence: parse the element type.
    if (p.parseComma())
    {
      p.emitError(p.getCurrentLocation(), "expected `,`");
      return {};
    }

    ::mlir::Type elementType;
    if (p.parseType(elementType))
    {
      p.emitError(p.getCurrentLocation(), "expected element type");
      return {};
    }

    if (p.parseGreater())
    {
      p.emitError(p.getCurrentLocation(), "expected `>`");
      return {};
    }

    if (ptrType.setElementType(elementType).failed())
    {
      p.emitError(p.getCurrentLocation(), "failed to complete deferred pointer type");
      return {};
    }
    return ptrType;
  }

  // Normal pointer with a concrete element type.
  ::mlir::Type T;
  if (p.parseType(T))
  {
    p.emitError(p.getCurrentLocation(), "expected element type");
    return {};
  }
  if (p.parseGreater())
  {
    p.emitError(p.getCurrentLocation(), "expected `>`");
    return {};
  }
  return PointerType::get(p.getContext(), T);
}

void PointerType::print(::mlir::AsmPrinter& p) const
{
  const auto id = this->getImpl()->m_id;
  const auto elementType = this->getElementType();

  if (id.empty() && !elementType)
  {
    // unsafe.Pointer: !go.ptr (no angle brackets)
    return;
  }

  p << "<";
  if (this->isDeferred())
  {
    p.printKeywordOrString(id);
    if (const auto cyclicPrint = p.tryStartCyclicPrint(*this);
        succeeded(cyclicPrint) && elementType)
    {
      // First occurrence: print the element type after the id.
      p << ", ";
      p.printType(*elementType);
    }
    // If cyclicPrint failed, we are in a cycle: only the id was printed,
    // which serves as the back-reference.
  }
  else
  {
    // Normal (non-deferred) pointer: just print the element type.
    p.printType(*elementType);
  }
  p << ">";
}

::llvm::TypeSize PointerType::getTypeSizeInBits(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
}

::llvm::TypeSize PointerType::getTypeSize(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  return dataLayout.getTypeSize(IndexType::get(this->getContext()));
}

uint64_t PointerType::getABIAlignment(const DataLayout& dataLayout, DataLayoutEntryListRef params)
  const
{
  return dataLayout.getTypeABIAlignment(IndexType::get(this->getContext()));
}

uint64_t PointerType::getPreferredAlignment(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
}

} // namespace mlir::go

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::go::PointerType)
