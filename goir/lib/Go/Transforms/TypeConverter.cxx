#include "Go/Transforms/TypeConverter.h"

#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "Go/Util.h"
#include <Go/IR/GoOps.h>

// CIR type headers for CGo type conversion support.
#include <clang/CIR/Dialect/IR/CIRTypes.h>

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
  this->ignoreType<mlir::IntegerType>();

  addSourceMaterialization(
    [&](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> mlir::Value
    {
      if (inputs.size() != 1)
      {
        return {};
      }

      // Handle integer type mismatch between dialects.
      return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
    });

  addTargetMaterialization(
    [&](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> mlir::Value
    {
      if (inputs.size() != 1)
      {
        return {};
      }

      // Handle integer type mismatch between dialects.
      return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
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

        // Check if this is a CGo struct (e.g. "main._cgo_testStruct").
        // Match the CIR naming convention ("struct.<cname>") so both sides
        // lower to the same LLVM struct type.
        auto id = type.getId().str();
        auto dotPos = id.rfind('.');
        if (dotPos != std::string::npos)
        {
          auto localName = llvm::StringRef(id).substr(dotPos + 1);
          if (localName.starts_with("_cgo_"))
            os << "struct." << localName.drop_front(5);
          else
            os << "llvm_struct_" << id;
        }
        else
        {
          os << "llvm_struct_" << id;
        }

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

  this->addConversion(
    [&](mlir::go::FunctionType T)
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

      const auto funcType = mlir::FunctionType::get(T.getContext(), inputTypes, resultTypes);
      SignatureConversion result(funcType.getNumInputs());
      return this->convertFunctionSignature(funcType, false, false, result);
    });

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

  // CIR type conversions for CGo support.
  // These mirror clang/CIR's prepareTypeConverter so that CIR types
  // encountered in go.bitcast ops can be converted during GoIR→LLVM lowering.
  mlir::DataLayout cirDataLayout(module);

  this->addConversion([&](cir::PointerType type) -> mlir::Type {
    mlir::ptr::MemorySpaceAttrInterface addrSpaceAttr = type.getAddrSpace();
    unsigned numericAS = 0;
    if (auto targetAsAttr =
            mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                addrSpaceAttr))
      numericAS = targetAsAttr.getValue();
    return mlir::LLVM::LLVMPointerType::get(type.getContext(), numericAS);
  });
  this->addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1);
  });
  this->addConversion([&](cir::IntType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  this->addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  this->addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  this->addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  this->addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  this->addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return this->convertType(type.getUnderlying());
  });
  this->addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  this->addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
  this->addConversion([&](cir::VoidType type) -> mlir::Type {
    return mlir::LLVM::LLVMVoidType::get(type.getContext());
  });
  this->addConversion([&](cir::ArrayType type) -> mlir::Type {
    // For CIR BoolType in arrays, use i8 (matching CIR's convertTypeForMemory).
    mlir::Type elemTy;
    if (isa<cir::BoolType>(type.getElementType()))
      elemTy = mlir::IntegerType::get(
          type.getContext(), cirDataLayout.getTypeSizeInBits(type.getElementType()));
    else
      elemTy = this->convertType(type.getElementType());
    return mlir::LLVM::LLVMArrayType::get(elemTy, type.getSize());
  });
  this->addConversion([&](cir::FuncType type) -> std::optional<mlir::Type> {
    auto result = this->convertType(type.getReturnType());
    llvm::SmallVector<mlir::Type> arguments;
    if (this->convertTypes(type.getInputs(), arguments).failed())
      return std::nullopt;
    return mlir::LLVM::LLVMFunctionType::get(result, arguments, type.isVarArg());
  });
  this->addConversion([&](cir::RecordType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> llvmMembers;
    switch (type.getKind())
    {
    case cir::RecordType::Class:
    case cir::RecordType::Struct:
      for (mlir::Type ty : type.getMembers())
      {
        if (isa<cir::BoolType>(ty))
          llvmMembers.push_back(mlir::IntegerType::get(
              ty.getContext(), cirDataLayout.getTypeSizeInBits(ty)));
        else
          llvmMembers.push_back(this->convertType(ty));
      }
      break;
    case cir::RecordType::Union:
      if (!type.getMembers().empty())
      {
        if (auto largestMember = type.getLargestMember(cirDataLayout))
        {
          if (isa<cir::BoolType>(largestMember))
            llvmMembers.push_back(mlir::IntegerType::get(
                largestMember.getContext(),
                cirDataLayout.getTypeSizeInBits(largestMember)));
          else
            llvmMembers.push_back(this->convertType(largestMember));
        }
        if (type.getPadded())
        {
          auto last = *type.getMembers().rbegin();
          if (isa<cir::BoolType>(last))
            llvmMembers.push_back(mlir::IntegerType::get(
                last.getContext(), cirDataLayout.getTypeSizeInBits(last)));
          else
            llvmMembers.push_back(this->convertType(last));
        }
      }
      break;
    }

    mlir::LLVM::LLVMStructType llvmStruct;
    if (type.getName())
    {
      llvmStruct = mlir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getPrefixedName());
      if (llvmStruct.setBody(llvmMembers, type.getPacked()).failed())
        llvm_unreachable("Failed to set body of CIR record");
    }
    else
    {
      llvmStruct = mlir::LLVM::LLVMStructType::getLiteral(
          type.getContext(), llvmMembers, type.getPacked());
    }
    return llvmStruct;
  });
}

mlir::Type LLVMTypeConverter::convertPointer(PointerType T) const
{
  return mlir::LLVM::LLVMPointerType::get(T.getContext());
}
} // namespace mlir::go
