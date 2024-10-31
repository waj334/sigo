
#include "Go/IR/Types/Interface.h"

#include <utility>

#include "Go/IR/GoTypes.h"
#include "llvm/Support/TypeSize.h"

namespace mlir::go
{
namespace detail
{
InterfaceTypeStorage::InterfaceTypeStorage(const std::string& name)
  : m_name(name)
{
}

InterfaceTypeStorage::InterfaceTypeStorage(const FunctionMap& methods)
  : m_methods(methods)
{
}

InterfaceTypeStorage::KeyTy InterfaceTypeStorage::getAsKey() const
{
  if (!this->m_name.empty())
  {
    return Key(this->getName());
  }
  else
  {
    return Key(this->getMethods());
  }
}

bool InterfaceTypeStorage::operator==(const KeyTy& key) const
{
  return (this->getAsKey().hashKey() == key.hashKey());
}

::llvm::hash_code InterfaceTypeStorage::hashKey(const KeyTy& key)
{
  return key.hashKey();
}

InterfaceTypeStorage* InterfaceTypeStorage::construct(
  ::mlir::TypeStorageAllocator& allocator,
  const KeyTy& key)
{
  if (!key.m_name.empty())
  {
    return new (allocator.allocate<InterfaceTypeStorage>()) InterfaceTypeStorage(key.m_name);
  }
  else
  {
    return new (allocator.allocate<InterfaceTypeStorage>()) InterfaceTypeStorage(key.m_methods);
  }
}

LogicalResult InterfaceTypeStorage::mutate(
  TypeStorageAllocator& allocator,
  const InterfaceTypeStorage::FunctionMap& methods)
{
  if (this->m_name.empty())
  {
    return failure();
  }

  if (this->m_isSet)
  {
    return success(this->getMethods() == methods);
  }

  // Check methods.
  for (const auto& [_, methodType] : methods)
  {
    if (!mlir::isa<FunctionType>(methodType))
    {
      return failure();
    }
  }

  this->m_methods = methods;
  this->m_isSet = true;
  return success();
}

std::string InterfaceTypeStorage::getName() const
{
  return this->m_name;
}

InterfaceTypeStorage::FunctionMap InterfaceTypeStorage::getMethods() const
{
  return this->m_methods;
}
} // namespace detail

InterfaceType InterfaceType::get(
  ::mlir::MLIRContext* context,
  const detail::InterfaceTypeStorage::FunctionMap& methods)
{
  return Base::get(context, methods);
}

InterfaceType InterfaceType::getNamed(::mlir::MLIRContext* context, const std::string& name)
{
  return Base::get(context, name);
}

::mlir::Type InterfaceType::parse(::mlir::AsmParser& p)
{
  InterfaceType result;
  std::string _name;
  detail::InterfaceTypeStorage::FunctionMap _methods;
  FailureOr<::mlir::AsmParser::CyclicParseReset> cyclicParse;

  //        !go.interface<any>
  //        !go.interface<{
  //            "foo" = (string,i32) -> (i32,f32),
  //            "bar" = (i16) -> (i64,string)
  //        }>
  //        !go.interface<"foobar">
  //        !go.interface<"myInterface", {
  //            "foo" = (string,i32) -> (i32,f32),
  //            "bar" = (i16) -> (i64,string),
  //            "recursificator" = (!go.interface<"myInterface">) -> (i32)
  //        }>

  if (p.parseLess())
  {
    p.emitError(p.getCurrentLocation(), "expected `<`");
    return {};
  }

  if (succeeded(p.parseOptionalString(&_name)))
  {
    // Go ahead and create the named interface type
    result = InterfaceType::getNamed(p.getContext(), _name);

    // Named interfaces can be cyclic. Deal with parsing this
    cyclicParse = p.tryStartCyclicParse(result);
    if (failed(cyclicParse))
    {
      // Don't parse any further
      if (p.parseGreater())
      {
        p.emitError(p.getCurrentLocation(), "expected `>`");
        return {};
      }
      return result;
    }
    else
    {
      if (p.parseComma())
      {
        p.emitError(p.getCurrentLocation(), "expected `,`");
        return {};
      }
    }
  }

  // Parse the optional `any` keyword
  if (failed(p.parseOptionalKeyword("any")))
  {
    // Optional methods start with `{`
    if (p.parseLBrace())
    {
      p.emitError(p.getCurrentLocation(), "expected `{`");
      return {};
    }

    // Parse optional methods
    while (true)
    {
      std::string methodName;
      FunctionType signature;

      // Parse the method name
      if (succeeded(p.parseOptionalString(&methodName)))
      {
        if (methodName.empty())
        {
          p.emitError(p.getCurrentLocation(), "method name cannot be empty string");
          return {};
        }

        if (p.parseEqual())
        {
          p.emitError(p.getCurrentLocation(), "expected `=`");
          return {};
        }

        // Parse the function signature
        if (p.parseType(signature))
        {
          p.emitError(p.getCurrentLocation(), "expected signature");
          return {};
        }

        if (_methods.find(methodName) != _methods.end())
        {
          p.emitError(
            p.getCurrentLocation(), "method called \"" + methodName + " appears more than once");
          return {};
        }
        _methods[methodName] = signature;
      }

      // Continue parsing the method signatures if there is a trailing comma
      if (failed(p.parseOptionalComma()))
      {
        // else stop
        break;
      }
    }

    if (p.parseRBrace())
    {
      p.emitError(p.getCurrentLocation(), "expected `}`");
      return {};
    }
  }

  if (p.parseGreater())
  {
    p.emitError(p.getCurrentLocation(), "expected `>`");
    return {};
  }

  if (!_name.empty())
  {
    if (failed(result.setMethods(_methods)))
    {
      return {};
    }
  }
  else
  {
    result = InterfaceType::get(p.getContext(), _methods);
  }
  return result;
}

void InterfaceType::print(::mlir::AsmPrinter& p) const
{
  //        !go.interface<any>
  //        !go.interface<{
  //            "foo" = (string,i32) -> (i32,f32),
  //            "bar" = (i16) -> (i64,string)
  //        }>
  //        !go.interface<"named", {
  //            "foo" = (string,i32) -> (i32,f32),
  //            "bar" = (i16) -> (i64,string),
  //            "foobar" = (!go.interface<"named">)
  //        }>

  const auto methods = this->getMethods();
  const auto name = this->getName();
  auto cyclicPrint = p.tryStartCyclicPrint(*this);
  p << "<";

  if (!name.empty())
  {
    p.printString(name);
    if (succeeded(cyclicPrint))
    {
      p << ", ";
    }
  }

  if (!methods.empty())
  {
    if (succeeded(cyclicPrint))
    {
      p << "{";
      auto it = methods.begin();
      while (it != methods.end())
      {
        p << "\"" << it->first << "\""
          << " = ";
        p.printStrippedAttrOrType(it->second);

        it++;
        if (it != methods.end())
        {
          p << ", ";
        }
      }
      p << "}";
    }
  }
  else
  {
    p << "any";
  }
  p << ">";
}

mlir::LogicalResult InterfaceType::setMethods(
  const detail::InterfaceTypeStorage::FunctionMap& methods)
{
  return Base::mutate(methods);
}

detail::InterfaceTypeStorage::FunctionMap InterfaceType::getMethods() const
{
  return getImpl()->getMethods();
}

std::string InterfaceType::getName() const
{
  return getImpl()->getName();
}

::llvm::TypeSize InterfaceType::getTypeSizeInBits(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  // The interface type consists of two pointers
  return dataLayout.getTypeSizeInBits(IndexType::get(getContext())) * 2;
}

::llvm::TypeSize InterfaceType::getTypeSize(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  // The interface type consists of two pointers
  return dataLayout.getTypeSize(IndexType::get(getContext())) * 2;
}

uint64_t InterfaceType::getABIAlignment(const DataLayout& dataLayout, DataLayoutEntryListRef params)
  const
{
  return dataLayout.getTypeABIAlignment(IndexType::get(getContext()));
}

uint64_t InterfaceType::getPreferredAlignment(
  const DataLayout& dataLayout,
  DataLayoutEntryListRef params) const
{
  return dataLayout.getTypePreferredAlignment(IndexType::get(getContext()));
}
} // namespace mlir::go

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::go::InterfaceType)
