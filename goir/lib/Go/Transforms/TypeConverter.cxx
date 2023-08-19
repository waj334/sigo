#include "Go/Transforms/TypeConverter.h"

#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>

#include "Go/Util.h"

#include <Go/IR/GoOps.h>

namespace mlir::go {
RuntimeTypeLookUp::RuntimeTypeLookUp(mlir::ModuleOp module) {
  assert(module->hasAttr("go.runtimeTypes"));
  auto typeMap = mlir::dyn_cast<mlir::DictionaryAttr>(module->getAttr("go.runtimeTypes"));
  for (auto entry : typeMap) {
    std::string mnemonic = entry.getName().getValue().str();
    m_typeMap[mnemonic] = mlir::dyn_cast<mlir::TypeAttr>(entry.getValue()).getValue();
  }
}

CoreTypeConverter::CoreTypeConverter(mlir::ModuleOp module) : TypeConverter(), RuntimeTypeLookUp(module) {
  mlir::DataLayout dataLayout(module);
  const auto indexTypeWidth = dataLayout.getTypeSizeInBits(IndexType::get(module->getContext()));

  // Add Go-specific type conversions
  this->addConversion([&](NamedType T) { return this->convertType(baseType(T)); });

  this->addConversion([&](FunctionType T) {
    TypeConverter::SignatureConversion result(T.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(convertSignatureArgs(T.getInputs(), result)) || failed(convertTypes(T.getResults(), newResults))) {
      return std::optional<Type>();
    }

    return std::optional<Type>(FunctionType::get(T.getContext(), result.getConvertedTypes(), newResults));
  });

  this->addConversion([&](IntegerType T) { return IntegerType::get(T.getContext(), T.getWidth()); });
  this->addConversion([indexTypeWidth](IntType T) { return IntegerType::get(T.getContext(), indexTypeWidth); });
  this->addConversion([indexTypeWidth](UintType T) { return IntegerType::get(T.getContext(), indexTypeWidth); });
  this->addConversion([indexTypeWidth](UintptrType T) { return IntegerType::get(T.getContext(), indexTypeWidth); });

  this->ignoreType<FloatType>();
  this->ignoreType<ArrayType>();
  this->ignoreType<ChanType>();
  this->ignoreType<InterfaceType>();
  this->ignoreType<MapType>();
  this->ignoreType<SliceType>();
  this->ignoreType<StringType>();
  this->ignoreType<PointerType>();
  this->ignoreType<LLVM::LLVMStructType>();
  this->ignoreType<ComplexType>();

  auto addBitcast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    auto cast = builder.create<mlir::go::BitcastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult());
  };

  addSourceMaterialization(addBitcast);
  addTargetMaterialization(addBitcast);

  /*
  // Use UnrealizedConversionCast as the bridge so that we don't need to pull
  // in patterns for other dialects.
  auto addUnrealizedCast = [this](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return cast.getResult(0);
  };

  addSourceMaterialization(addUnrealizedCast);
  addTargetMaterialization(addUnrealizedCast);
  */
}

mlir::Type CoreTypeConverter::convertArray(ArrayType T) const {
  // Convert the element type
  auto ET = this->convertType(T.getElementType());

  // Create the equivalent LLVM array type
  return ::mlir::LLVM::LLVMArrayType::get(ET, T.getLength());
}

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp module, const mlir::LowerToLLVMOptions &options)
    : mlir::LLVMTypeConverter(module.getContext(), options), RuntimeTypeLookUp(module) {
  this->addConversion([&](NamedType T) { return this->convertType(baseType(T)); });

  this->addConversion([&](ArrayType T) {
    // Convert the element type
    auto ET = this->convertType(T.getElementType());

    // Create the equivalent LLVM array type
    return ::mlir::LLVM::LLVMArrayType::get(ET, T.getLength());
  });

  this->addConversion([&](IntType T) { return IntegerType::get(T.getContext(), this->getIndexTypeBitwidth()); });
  this->addConversion([&](UintType T) { return IntegerType::get(T.getContext(), this->getIndexTypeBitwidth()); });
  this->addConversion([&](UintptrType T) { return IntegerType::get(T.getContext(), this->getIndexTypeBitwidth()); });
  this->addConversion([&](PointerType T) { return this->convertPointer(T); });
  this->addRuntimeTypeConversion<ChanType>();
  this->addRuntimeTypeConversion<InterfaceType>();
  this->addRuntimeTypeConversion<MapType>();
  this->addRuntimeTypeConversion<SliceType>();
  this->addRuntimeTypeConversion<StringType>();
}

mlir::Type LLVMTypeConverter::convertPointer(PointerType T) const {
  return mlir::LLVM::LLVMPointerType::get(T.getContext());
}
} // namespace mlir::go
