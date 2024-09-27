#include "Go/Transforms/TypeConverter.h"

#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "Go/Util.h"
#include <Go/IR/GoOps.h>

namespace mlir::go
{
RuntimeTypeLookUp::RuntimeTypeLookUp(mlir::ModuleOp module)
{
  assert(module->hasAttr("go.runtimeTypes"));
  auto typeMap = mlir::dyn_cast<mlir::DictionaryAttr>(module->getAttr("go.runtimeTypes"));
  for (auto entry : typeMap)
  {
    std::string mnemonic = entry.getName().getValue().str();
    m_typeMap[mnemonic] = mlir::dyn_cast<mlir::TypeAttr>(entry.getValue()).getValue();
  }
}

CoreTypeConverter::CoreTypeConverter(mlir::ModuleOp module)
  : TypeConverter()
  , RuntimeTypeLookUp(module)
{
  mlir::DataLayout dataLayout(module);
  const uint64_t indexTypeWidth = dataLayout.getTypeSizeInBits(IndexType::get(module.getContext()));

  // Add Go-specific type conversions
  this->addConversion([&](NamedType T) { return this->convertType(baseType(T)); });

  this->addConversion(
    [&](FunctionType T)
    {
      TypeConverter::SignatureConversion result(T.getNumInputs());
      SmallVector<Type, 1> newResults;
      if (
        failed(convertSignatureArgs(T.getInputs(), result)) ||
        failed(convertTypes(T.getResults(), newResults)))
      {
        return std::optional<Type>();
      }

      return std::optional<Type>(
        FunctionType::get(T.getContext(), result.getConvertedTypes(), newResults));
    });

  this->addConversion([&](BooleanType T) { return mlir::IntegerType::get(T.getContext(), 1); });

  this->addConversion(
    [indexTypeWidth](IntegerType T)
    {
      if (auto width = T.getWidth(); width.has_value())
      {
        return mlir::IntegerType::get(T.getContext(), *width);
      }
      return mlir::IntegerType::get(T.getContext(), indexTypeWidth);
    });

  this->addConversion(
    [&](FunctionType T)
    {
      SmallVector<Type> inputTypes;
      inputTypes.reserve(T.hasReceiver() ? T.getNumInputs() + 1 : T.getNumInputs());

      SmallVector<Type> resultTypes;
      resultTypes.reserve(T.getNumResults());

      if (T.hasReceiver())
      {
        inputTypes.push_back(this->convertType(T.getReceiver()));
      }

      for (size_t i = 0; i < T.getNumInputs(); ++i)
      {
        inputTypes.push_back(this->convertType(T.getInput(i)));
      }

      for (size_t i = 0; i < T.getNumResults(); ++i)
      {
        resultTypes.push_back(this->convertType(T.getResult(i)));
      }

      return mlir::FunctionType::get(T.getContext(), inputTypes, resultTypes);
    });

  this->ignoreType<FloatType>();
  this->ignoreType<ArrayType>();
  this->ignoreType<ChanType>();
  this->ignoreType<InterfaceType>();
  this->ignoreType<MapType>();
  this->ignoreType<SliceType>();
  this->ignoreType<StringType>();
  this->ignoreType<PointerType>();
  this->ignoreType<GoStructType>();
  this->ignoreType<LLVM::LLVMStructType>();
  this->ignoreType<ComplexType>();

  addSourceMaterialization(
    [&](
      OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> std::optional<Value>
    {
      if (inputs.size() != 1)
      {
        return std::nullopt;
      }

      /*
      if (mlir::isa<mlir::IntegerType>(resultType))
      {*/
        // Handle integer type mismatch between dialects.
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      /*}

      // Otherwise, this type can be bitcasted using the Go dialect's bitcast operation.
      return builder.create<mlir::go::BitcastOp>(loc, resultType, inputs);*/
    });

  addTargetMaterialization(
    [&](
      OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> std::optional<Value>
    {
      if (inputs.size() != 1)
      {
        return std::nullopt;
      }

      /*
      if (mlir::go::isa<IntegerType>(resultType))
      {*/
        // Handle integer type mismatch between dialects.
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      /*}

      // Otherwise, this type can be bitcasted using the Go dialect's bitcast operation.
      return builder.create<mlir::go::BitcastOp>(loc, resultType, inputs);*/
    });
}

mlir::Type CoreTypeConverter::convertArray(ArrayType T) const
{
  // Convert the element type
  auto ET = this->convertType(T.getElementType());

  // Create the equivalent LLVM array type
  return ::mlir::LLVM::LLVMArrayType::get(ET, T.getLength());
}

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp module, const mlir::LowerToLLVMOptions& options)
  : mlir::LLVMTypeConverter(module.getContext(), options)
  , RuntimeTypeLookUp(module)
{
  this->addConversion([&](const NamedType T) { return this->convertType(baseType(T)); });

  this->addConversion(
    [&](const ArrayType T)
    {
      // Convert the element type
      const auto ET = this->convertType(T.getElementType());

      // Create the equivalent LLVM array type
      return ::mlir::LLVM::LLVMArrayType::get(ET, T.getLength());
    });

  this->addConversion(
    [&](const GoStructType type) -> std::optional<Type>
    {
      const auto hashCode = llvm::hash_combine(type);
      if (const auto it = this->m_typeMap.find(hashCode); it != this->m_typeMap.end())
      {
        // Return the cached type.
        return it->second;
      }

      mlir::LLVM::LLVMStructType structType;
      if (!type.isLiteral())
      {
        std::string name;
        llvm::raw_string_ostream os(name);
        os << "llvm_struct_" << type.getId();
        structType = mlir::LLVM::LLVMStructType::getIdentified(type.getContext(), name);

        // Cache this struct type for converting recursive struct types.
        this->m_typeMap[hashCode] = structType;
      }

      // Convert the struct field types.
      SmallVector<Type> fieldTypes;
      fieldTypes.reserve(type.getNumFields());
      if (this->convertTypes(type.getFieldTypes(), fieldTypes).failed())
      {
        return std::nullopt;
      }

      if (type.isLiteral())
      {
        return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
      }

      if (structType.setBody(fieldTypes, false).failed())
      {
        return std::nullopt;
      }
      return structType;
    });

  /*
  this->addConversion([&](FunctionType T) {
    SmallVector<Type> inputTypes;
    inputTypes.reserve(T.hasReceiver() ? T.getNumInputs()+1 : T.getNumInputs());

    SmallVector<Type> resultTypes;
    resultTypes.reserve(T.getNumResults());

    if (T.hasReceiver())
    {
      inputTypes.push_back(this->convertType(T.getReceiver()));
    }

    for (size_t i = 0; i < T.getNumInputs(); ++i)
    {
      inputTypes.push_back(this->convertType(T.getInput(i)));
    }

    for (size_t i = 0; i < T.getNumResults(); ++i)
    {
      resultTypes.push_back(this->convertType(T.getResult(i)));
    }

    return mlir::FunctionType::get(T.getContext(), inputTypes, resultTypes);
  });*/

  this->addConversion([&](BooleanType T) { return mlir::IntegerType::get(T.getContext(), 1); });

  this->addConversion(
    [&](IntegerType T)
    {
      if (auto width = T.getWidth(); width.has_value())
      {
        return mlir::IntegerType::get(T.getContext(), *width);
      }
      return mlir::IntegerType::get(T.getContext(), this->getPointerBitwidth());
    });
  this->addConversion([&](PointerType T) { return this->convertPointer(T); });
  this->addRuntimeTypeConversion<ChanType>();
  this->addRuntimeTypeConversion<InterfaceType>();
  this->addRuntimeTypeConversion<MapType>();
  this->addRuntimeTypeConversion<SliceType>();
  this->addRuntimeTypeConversion<StringType>();
}

mlir::Type LLVMTypeConverter::convertPointer(PointerType T) const
{
  return mlir::LLVM::LLVMPointerType::get(T.getContext());
}
} // namespace mlir::go
