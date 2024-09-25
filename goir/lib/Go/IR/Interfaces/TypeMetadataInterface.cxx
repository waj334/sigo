#include "Go/IR/GoDialect.h"
#include <Go/Util.h>

namespace mlir::go
{

mlir::DictionaryAttr TypeMetadataInterface::getTypeMetadata(mlir::Type type)
{
  {
    llvm::sys::SmartScopedReader<true> typeLock(this->m_mutex);
    if (auto it = this->m_metadataCache.find(type); it != this->m_metadataCache.end())
    {
      return it->second;
    }
  }

  {
    llvm::sys::SmartScopedWriter<true> typeLock(this->m_mutex);
    // Double-check that no other thread has already created the metadata since this critical
    // section can be attempted by multiple threads at once.
    if (auto it = this->m_metadataCache.find(type); it != this->m_metadataCache.end())
    {
      return it->second;
    }

    mlir::DictionaryAttr result;
    mlir::SmallVector<mlir::NamedAttribute> entries;
    entries.push_back(
      this->namedOf("typeCode", this->intAttrOf(static_cast<int64_t>(GetGoTypeId(type)))));
    entries.push_back(this->namedOf("typeHash", this->intAttrOf(llvm::hash_combine(type))));

    // Generate the respective type information for the input type.
    llvm::TypeSwitch<mlir::Type>(type)
      .Case(
        [&](ArrayType type)
        {
          entries.push_back(this->namedOf("elementType", this->typeAttrOf(type.getElementType())));
          entries.push_back(this->namedOf("length", this->intAttrOf(type.getLength())));
        })
      .Case(
        [&](ChanType type)
        {
          entries.push_back(this->namedOf("elementType", this->typeAttrOf(type.getElementType())));
          entries.push_back(
            this->namedOf("direction", this->intAttrOf(static_cast<int64_t>(type.getDirection()))));
        });

    if (!entries.empty())
    {
      result = mlir::DictionaryAttr::get(this->getContext(), entries);
    }

    // Cache and return the result.
    this->m_metadataCache[type] = result;
    return result;
  }
}

mlir::NamedAttribute TypeMetadataInterface::namedOf(mlir::StringRef name, mlir::Attribute attr)
{
  return { mlir::StringAttr::get(this->getContext(), name), attr };
}

mlir::TypeAttr TypeMetadataInterface::typeAttrOf(mlir::Type type)
{
  return mlir::TypeAttr::get(type);
}

mlir::IntegerAttr TypeMetadataInterface::intAttrOf(int64_t value)
{
  return mlir::IntegerAttr::get(mlir::IntegerType::get(this->getContext(), 64), value);
}

} // namespace mlir::go
