#pragma once

#include <filesystem>

#include <llvm/BinaryFormat/Dwarf.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"

namespace mlir::go
{

struct BaseAttachDebugInfoPass
{
  DenseMap<Type, LLVM::DITypeAttr> m_typeMap;
  DenseMap<Type, DistinctAttr> m_idMap;
  DenseMap<Type, DictionaryAttr> m_typeDataMap;
  DenseMap<Type, LocationAttr> m_typeDeclareLocationMap;

  LLVM::DITypeAttr getDITypeAttr(
    MLIRContext* context,
    Type type,
    const DataLayout& dataLayout,
    const RuntimeTypeLookUp& runtimeTypes,
    const StringRef name = StringRef())
  {
    if (m_typeMap.lookup(type))
    {
      return m_typeMap[type];
    }

    // Clean up name.
    std::string _name = name.str();
    stringReplaceAll(_name, ".", "_");
    stringReplaceAll(_name, "/", "_");
    stringReplaceAll(_name, "-", "_");

    LLVM::DITypeAttr result;

    const auto kind = GetGoTypeId(baseType(type));
    const unsigned size = dataLayout.getTypeSizeInBits(type);
    const unsigned align = dataLayout.getTypeABIAlignment(type);

    if (const auto namedType = mlir::dyn_cast<NamedType>(type); namedType)
    {
      const auto underlyingType = getDITypeAttr(
        context, namedType.getUnderlying(), dataLayout, runtimeTypes, namedType.getName());

      std::string alias = namedType.getName().str();
      stringReplaceAll(alias, ".", "_");
      stringReplaceAll(alias, "/", "_");
      stringReplaceAll(alias, "-", "_");

      return LLVM::DIDerivedTypeAttr::get(
        context,
        llvm::dwarf::DW_TAG_typedef,
        mlir::StringAttr::get(context, alias),
        underlyingType,
        size,
        align,
        0,
        std::nullopt,
        LLVM::DINodeAttr());
    }

    // Create the type information based kind
    switch (kind)
    {
      case GoTypeId::Bool:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "bool", 1, llvm::dwarf::DW_ATE_boolean);
        break;
      case GoTypeId::Int:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "int", size, llvm::dwarf::DW_ATE_signed);
        break;
      case GoTypeId::Int8:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "char", size, llvm::dwarf::DW_ATE_signed);
        break;
      case GoTypeId::Int16:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "short", size, llvm::dwarf::DW_ATE_signed);
        break;
      case GoTypeId::Int32:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "int", size, llvm::dwarf::DW_ATE_signed);
        break;
      case GoTypeId::Int64:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "long long int",
          size,
          llvm::dwarf::DW_ATE_signed);
        break;
      case GoTypeId::Uint:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned int",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Uint8:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned char",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Uint16:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned short",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Uint32:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned int",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Uint64:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned long long int",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Uintptr:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "unsigned long",
          size,
          llvm::dwarf::DW_ATE_unsigned);
        break;
      case GoTypeId::Float32:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "float", size, llvm::dwarf::DW_ATE_float);
        break;
      case GoTypeId::Float64:
        result = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "double", size, llvm::dwarf::DW_ATE_float);
        break;
      case GoTypeId::Complex64:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "complex64",
          size,
          llvm::dwarf::DW_ATE_complex_float);
        break;
      case GoTypeId::Complex128:
        result = LLVM::DIBasicTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_base_type,
          "complex128",
          size,
          llvm::dwarf::DW_ATE_complex_float);
        break;
      case GoTypeId::Array:
      {
        const auto recId = this->getOrCreateId(type);
        const auto arrayType = go::cast<ArrayType>(type);
        const auto lengthAttr =
          IntegerAttr::get(mlir::IntegerType::get(context, 64), arrayType.getLength());
        const auto sizeAttr =
          IntegerAttr::get(mlir::IntegerType::get(context, 64), arrayType.getLength());
        const auto diSubrange =
          LLVM::DISubrangeAttr::get(context, lengthAttr, 0, IntegerAttr(), sizeAttr);

        result = mlir::LLVM::DICompositeTypeAttr::get(
          /*context=*/context, /*recId=*/recId,  /*isRecSelf=*/false, llvm::dwarf::DW_TAG_array_type,
          /*name=*/StringAttr::get(context, _name),
          /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr,
          /*baseType=*/getDITypeAttr(context, arrayType.getElementType(), dataLayout, runtimeTypes),
          /*flags=*/mlir::LLVM::DIFlags::Zero, /*sizeInBits=*/size, /*alignInBits=*/align,
          /*dataLocation=*/nullptr, /*rank=*/nullptr,
          /*allocated=*/nullptr, /*associated=*/nullptr, /*elements*/{ diSubrange });
        break;
      }
      case GoTypeId::Chan:
        result =
          getDITypeAttr(context, runtimeTypes.lookupRuntimeType("chan"), dataLayout, runtimeTypes);
        break;
      case GoTypeId::Func:
      {
        const auto signature = go::cast<FunctionType>(type);
        SmallVector<LLVM::DITypeAttr, 10> argTypes;
        for (const auto& argType : signature.getInputs())
        {
          argTypes.push_back(getDITypeAttr(context, argType, dataLayout, runtimeTypes));
        }
        auto diSignature = LLVM::DISubroutineTypeAttr::get(context, argTypes);
        result = LLVM::DIDerivedTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_pointer_type,
          StringAttr::get(context, ""),
          diSignature,
          size,
          align,
          0,
          std::nullopt,
          LLVM::DINodeAttr());
        break;
      }
      case GoTypeId::Interface:
        result = getDITypeAttr(
          context, runtimeTypes.lookupRuntimeType("interface"), dataLayout, runtimeTypes);
        break;
      case GoTypeId::Map:
        result =
          getDITypeAttr(context, runtimeTypes.lookupRuntimeType("map"), dataLayout, runtimeTypes);
        break;
      case GoTypeId::Pointer:
      {
        const auto ptrType = go::cast<PointerType>(type);
        result = LLVM::DIDerivedTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_pointer_type,
          StringAttr::get(context, ""),
          getDITypeAttr(context, *ptrType.getElementType(), dataLayout, runtimeTypes),
          size,
          align,
          0,
          std::nullopt,
          LLVM::DINodeAttr());
        break;
      }
      case GoTypeId::Slice:
        result =
          getDITypeAttr(context, runtimeTypes.lookupRuntimeType("slice"), dataLayout, runtimeTypes);
        break;
      case GoTypeId::String:
        result = getDITypeAttr(
          context, runtimeTypes.lookupRuntimeType("string"), dataLayout, runtimeTypes);
        break;
      case GoTypeId::Struct:
      {
        const auto structT = mlir::cast<GoStructType>(type);
        const auto fields = structT.getFields();
        const auto numFields = fields.size();

        // Create the element types.
        SmallVector<LLVM::DINodeAttr> elementAttrs;
        const auto recId = this->getOrCreateId(type);

        // Short-circuit the type map so that the recursive self is returned for recursive types.
        this->m_typeMap[type] =
          mlir::cast<LLVM::DICompositeTypeAttr>(LLVM::DICompositeTypeAttr::getRecSelf(recId));

        for (size_t i = 0; i < numFields; i++)
        {
          const auto [fieldName, fieldType, fieldTags] = fields[i];
          const uint64_t fieldsSizeInBits = dataLayout.getTypeSizeInBits(fieldType);
          const uint64_t alignmentInBits = dataLayout.getTypeABIAlignment(fieldType) * 8;
          const auto offsetInBits = structT.getFieldOffset(dataLayout, i) * 8;

          auto fieldNameAttr = fieldName;
          if (!fieldNameAttr)
          {
            fieldNameAttr = StringAttr::get(context, "?");
          }

          // Skip fields named "_".
          if (fieldNameAttr.str() != "_")
          {
            const auto elementTypeAttr =
              getDITypeAttr(context, fieldType, dataLayout, runtimeTypes);
            const auto derivedTypeAttr = LLVM::DIDerivedTypeAttr::get(
              context,
              llvm::dwarf::DW_TAG_member,
              fieldNameAttr,
              elementTypeAttr,
              fieldsSizeInBits,
              alignmentInBits,
              offsetInBits,
              std::nullopt,
              LLVM::DINodeAttr());
            elementAttrs.push_back(derivedTypeAttr);
          }
        }

        // Align the total struct size to the maximum of the struct alignment and the largest field
        // alignment
        const uint64_t structAlignmentInBits = dataLayout.getTypeABIAlignment(structT);
        const uint64_t structSizeInBits = dataLayout.getTypeSizeInBits(structT);

        // Create the composite type.
        const auto nameAttr = StringAttr::get(context, _name);
        result = mlir::LLVM::DICompositeTypeAttr::get(
          /*context=*/context, /*recId=*/recId,  /*isRecSelf=*/false, llvm::dwarf::DW_TAG_structure_type,
          /*name=*/nameAttr, /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr,
          /*baseType=*/LLVM::DINullTypeAttr::get(context),
          /*flags=*/mlir::LLVM::DIFlags::Zero, /*sizeInBits=*/structSizeInBits,
          /*alignInBits=*/structAlignmentInBits, /*dataLocation=*/nullptr, /*rank=*/nullptr,
          /*allocated=*/nullptr, /*associated=*/nullptr, /*elements*/elementAttrs);
      }
      break;
      case GoTypeId::UnsafePointer:
      {
        const auto base = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "void", 0, llvm::dwarf::DW_ATE_unsigned);
        result = LLVM::DIDerivedTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_pointer_type,
          StringAttr::get(context, "void*"),
          base,
          size,
          align,
          0,
          std::nullopt,
          LLVM::DINodeAttr());
      }
      break;
      default:
        return {};
    }

    this->m_typeMap[type] = result;
    return result;
  }

  DistinctAttr getOrCreateId(Type type)
  {
    auto result = this->m_idMap[type];
    if (!result)
    {
      result = DistinctAttr::create(TypeAttr::get(type));
      m_idMap[type] = result;
    }
    return result;
  }

};

}