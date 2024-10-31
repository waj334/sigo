#include "../../include/Go/Util.h"

#include <mutex>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

namespace mlir::go
{

GoTypeId GetGoTypeId(const mlir::Type& type)
{
  return TypeSwitch<Type, GoTypeId>(type)
    .Case([](BooleanType T) { return GoTypeId::Bool; })
    .Case(
      [](IntegerType T)
      {
        if (T.isUintptr())
        {
          return GoTypeId::Uintptr;
        }

        if (auto width = T.getWidth(); width.has_value())
        {
          switch (*width)
          {
            case 8:
              return T.isSigned() ? GoTypeId::Int8 : GoTypeId::Uint8;
            case 16:
              return T.isSigned() ? GoTypeId::Int16 : GoTypeId::Uint16;
            case 32:
              return T.isSigned() ? GoTypeId::Int32 : GoTypeId::Uint32;
            case 64:
              return T.isSigned() ? GoTypeId::Int64 : GoTypeId::Uint64;
            default:
              return GoTypeId::Invalid;
          }
        }

        // This integer is not any fixed-width integer.
        return T.isSigned() ? GoTypeId::Int : GoTypeId::Uint;
      })
    .Case([](FloatType T) { return T.getWidth() == 32 ? GoTypeId::Float32 : GoTypeId::Float64; })
    .Case(
      [](ComplexType T)
      {
        return mlir::cast<FloatType>(T.getElementType()).getWidth() == 32 ? GoTypeId::Complex64
                                                                          : GoTypeId::Complex128;
      })
    .Case([](ArrayType T) { return GoTypeId::Array; })
    .Case([](ChanType T) { return GoTypeId::Chan; })
    .Case([](FunctionType T) { return GoTypeId::Func; })
    .Case([](InterfaceType T) { return GoTypeId::Interface; })
    .Case([](MapType T) { return GoTypeId::Map; })
    .Case([](PointerType T)
          { return T.getElementType().has_value() ? GoTypeId::Pointer : GoTypeId::UnsafePointer; })
    .Case([](SliceType T) { return GoTypeId::Slice; })
    .Case([](StringType T) { return GoTypeId::String; })
    .Case([](GoStructType T) { return GoTypeId::Struct; })
    .Default([](Type T) { return GoTypeId::Invalid; });
}

std::string typeStr(const mlir::Type& T)
{
  std::string typeStr;
  llvm::raw_string_ostream in(typeStr);
  T.print(in);
  return typeStr;
}

llvm::hash_code
computeMethodHash(const StringRef name, const FunctionType func, bool isInterface)
{
  llvm::hash_code result = llvm::hash_value(name);
  size_t offset = 0;
  if (!isInterface)
  {
    // Skip the receiver for named-type methods.
    offset = 1;
  }

  // Hash the input types starting at the offset.
  for (size_t i = offset; i < func.getNumInputs(); i++)
  {
    std::string str;
    llvm::raw_string_ostream(str) << func.getInput(i);
    result = llvm::hash_combine(result, str);
  }

  // Hash the result types
  for (auto t : func.getResults())
  {
    std::string str;
    llvm::raw_string_ostream(str) << t;
    result = llvm::hash_combine(result, str);
  }
  return result;
}

void stringReplaceAll(std::string& input, const std::string& substr, const std::string& str)
{
  for (size_t pos = input.find(substr); pos != std::string::npos; pos = input.find(substr))
  {
    input.replace(pos, substr.size(), str);
  }
}

//std::string formatPackageSymbol(std::string pkg, std::string symbol)
std::string formatPackageSymbol(const std::string& pkg, const std::string& symbol)
{
  /*
  // NOTE: Name mangling is disabled for now.
  stringReplaceAll(pkg, "/", "$");
  stringReplaceAll(symbol, ".", "@");
  return  pkg + "@" + symbol;
  */
  return pkg + "." + symbol;
}

} // namespace mlir::go
