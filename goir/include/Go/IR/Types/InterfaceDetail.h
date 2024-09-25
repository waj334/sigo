#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

namespace mlir::go
{

namespace detail
{
struct InterfaceTypeStorage : public ::mlir::TypeStorage
{
  using FunctionMap = std::map<std::string, Type>;

  struct Key
  {
    explicit Key(const std::string& name)
      : m_name(name)
    {
    }

    explicit Key(FunctionMap methods)
      : m_methods(std::move(methods))
    {
    }

    [[nodiscard]] ::llvm::hash_code hashKey() const
    {
      llvm::hash_code result{};
      if (!this->m_name.empty())
      {
        result = llvm::hash_value(this->m_name);
      }
      else
      {
        // Hash each field
        for (auto field : this->m_methods)
        {
          result = ::llvm::hash_combine(result, field);
        }
      }
      return result;
    }

    std::string m_name;
    FunctionMap m_methods;
  };

  using KeyTy = Key;

  explicit InterfaceTypeStorage(const std::string& name);

  explicit InterfaceTypeStorage(const FunctionMap& methods);

  bool operator==(const KeyTy& key) const;

  [[nodiscard]] KeyTy getAsKey() const;

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] FunctionMap getMethods() const;

  static ::llvm::hash_code hashKey(const KeyTy& key);

  static InterfaceTypeStorage* construct(::mlir::TypeStorageAllocator& allocator, const KeyTy& key);

  LogicalResult mutate(TypeStorageAllocator& allocator, const FunctionMap& methods);

  std::string m_name;
  FunctionMap m_methods;
  bool m_isSet = false;
};

} // namespace detail

}