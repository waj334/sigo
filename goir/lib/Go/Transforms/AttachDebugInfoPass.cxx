#include <filesystem>

#include <llvm/BinaryFormat/Dwarf.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"

namespace mlir::go
{

struct AttachDebugInfoPass : PassWrapper<AttachDebugInfoPass, OperationPass<ModuleOp>>
{
  DenseMap<Type, LLVM::DITypeAttr> m_typeMap;
  DenseMap<Type, DistinctAttr> m_idMap;
  DenseMap<Type, DictionaryAttr> m_typeDataMap;
  DenseMap<Type, LocationAttr> m_typeDeclareLocationMap;

  void runOnOperation() final
  {
    MLIRContext* context = &this->getContext();
    auto module = getOperation();

    auto runtimeTypes = RuntimeTypeLookUp(module);
    DataLayout dataLayout(module);

    // Create the builder.
    OpBuilder builder(module.getBodyRegion());

    // Collect all information about type declarations.
    module.walk(
      [&](DeclareTypeOp op)
      {
        const auto T = op.getDeclaredType();
        this->m_typeDataMap[T] = op->getAttrDictionary();
        this->m_typeDeclareLocationMap[T] = op->getLoc();
      });

    // Walk the module and create a subprogram for each function.
    module.walk(
      [&](mlir::go::FuncOp op)
      {
        if (op.getBody().empty())
        {
          // Skip forward declared functions.
          return;
        }

        // Functions that should generate debug information have a compile unit
        // fused with its location.
        if (const auto fusedLocWithCompileUnit =
              op->getLoc()->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
            fusedLocWithCompileUnit)
        {
          const auto compileUnitAttr = fusedLocWithCompileUnit.getMetadata();
          const auto funcNameAttr = op.getSymNameAttr();

          auto baseFuncName = funcNameAttr.strref();
          if (baseFuncName.contains("."))
          {
            // Extract the function's unqualified name.
            baseFuncName = baseFuncName.substr(baseFuncName.find_last_of(".") + 1);
          }
          auto baseFuncNameAttr = StringAttr::get(context, baseFuncName);

          const auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
          const auto filePath = std::filesystem::path(loc.getFilename().str());
          const auto fileName = filePath.filename().generic_string();
          const auto fileDir = filePath.parent_path().generic_string();
          const auto fileAttr = LLVM::DIFileAttr::get(context, fileName, fileDir);
          const auto subprogramIdAttr = DistinctAttr::create(UnitAttr::get(context));
          const auto subroutineTypeAttr =
            LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});
          const auto subprogramAttr = LLVM::DISubprogramAttr::get(
            context,
            subprogramIdAttr,
            compileUnitAttr,
            fileAttr,
            baseFuncNameAttr,
            funcNameAttr,
            fileAttr,
            /*line=*/loc.getLine(),
            /*scopeline=*/loc.getLine(),
            LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
            subroutineTypeAttr);
          op->setLoc(FusedLoc::get(context, { op.getLoc() }, subprogramAttr));
        }
      });

    // Walk the module and add metadata for globals.
    module.walk(
      [&](GlobalOp op)
      {
        const Location loc = op->getLoc();
        const auto fusedLocWithCompileUnit =
          loc->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();

        // Globals must have an associated compile unit in order for debug information about it to
        // be emitted.
        if (!fusedLocWithCompileUnit)
          return;

        const auto elementT = op.getGlobalType();
        const auto typeAttr = getDITypeAttr(context, elementT, dataLayout, runtimeTypes);

        const auto alignment = dataLayout.getTypePreferredAlignment(elementT);
        const auto compileUnitAttr = fusedLocWithCompileUnit.getMetadata();
        const auto linknameAttr = op.getSymNameAttr();

        std::string name = op.getSymNameAttr().str();
        if (const auto index = name.find('.'); index != std::string::npos)
        {
          name = name.substr(index + 1);
        }
        const auto nameAttr = StringAttr::get(context, name);

        const auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
        const auto filePath = std::filesystem::path(fileLoc.getFilename().str());
        const auto fileName = filePath.filename().generic_string();
        const auto fileDir = filePath.parent_path().generic_string();
        const auto fileAttr = LLVM::DIFileAttr::get(context, fileName, fileDir);
        const auto diGlobalAttr = LLVM::DIGlobalVariableAttr::get(
          context,
          compileUnitAttr,
          nameAttr,
          linknameAttr,
          fileAttr,
          fileLoc.getLine(),
          typeAttr,
          false,
          true,
          alignment);
        const auto diGlobalExprAttr = LLVM::DIGlobalVariableExpressionAttr::get(
          context, diGlobalAttr, LLVM::DIExpressionAttr());
        op->setLoc(FusedLoc::get(context, { op.getLoc() }, diGlobalExprAttr));
      });

    // Walk the module and attach debug info to all alloca operations.
    module.walk([&](AllocaOp op) { processAllocOp(op, builder, dataLayout, runtimeTypes); });
  }

  void processAllocOp(
    AllocaOp op,
    OpBuilder& builder,
    DataLayout& dataLayout,
    RuntimeTypeLookUp& runtimeTypes)
  {
    MLIRContext* context = op->getContext();
    const auto name = op.getVarName();

    // Get the allocated type that the debug information will be associated
    // with.
    const Type elementType = mlir::cast<TypeAttr>(op->getAttr("element")).getValue();

    // Not all allocs have debug information associated with them. Process
    // only the ones that do.
    if (name && !(*name).empty())
    {
      const auto fusedLoc =
        op->getParentOp()->getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>();
      if (!fusedLoc)
        return;

      const LLVM::DITypeAttr diType = getDITypeAttr(context, elementType, dataLayout, runtimeTypes);
      if (!diType)
        return;

      const auto scope = fusedLoc.getMetadata();
      const auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      const auto path = std::filesystem::path(loc.getFilename().str());
      const auto diFile =
        LLVM::DIFileAttr::get(context, path.filename().string(), path.parent_path().string());
      const auto diLocalVarAttr = LLVM::DILocalVariableAttr::get(
        scope,
        *name,
        diFile,
        loc.getLine(),                  // LINE
        0,                              // ARG,
        dataLayout.getStackAlignment(), // ALIGN
        diType);

      // Attach the debug information to the operation. The LLVM lowering pass
      // will actually create the required operations.
      op->setLoc(FusedLoc::get(context, { op.getLoc() }, diLocalVarAttr));
    }
  }

  LLVM::DITypeAttr getDITypeAttr(
    MLIRContext* context,
    Type type,
    const DataLayout& dataLayout,
    const RuntimeTypeLookUp& runtimeTypes,
    StringRef name = StringRef())
  {
    if (m_typeMap.lookup(type))
    {
      return m_typeMap[type];
    }

    LLVM::DITypeAttr result;

    const auto kind = GetGoTypeId(baseType(type));
    const unsigned size = dataLayout.getTypeSizeInBits(type);
    const unsigned align = dataLayout.getTypeABIAlignment(type);
    const auto extraData = this->m_typeDataMap[type];

    LLVM::DIScopeAttr diScope;
    LLVM::DIFileAttr diFile;

    if (const auto it = this->m_typeDeclareLocationMap.find(type);
        it != this->m_typeDeclareLocationMap.end())
    {
      auto [_, loc] = *it;
      const auto lineColLoc = loc.findInstanceOf<FileLineColLoc>();
      const auto filePath = std::filesystem::path(lineColLoc.getFilename().str());
      const auto fileName = filePath.filename().string();
      const auto fileDir = filePath.parent_path().string();
      diFile = LLVM::DIFileAttr::get(context, fileName, fileDir);

      if (const auto compileUnitLoc = loc.findInstanceOf<FusedLocWith<LLVM::DICompileUnitAttr>>())
      {
        diScope = compileUnitLoc.getMetadata();
      }
    }
    else
    {
      diScope = LLVM::DIFileAttr::get(context, "<unknown>", "<unknown>");
      diFile = LLVM::DIFileAttr::get(context, "<unknown>", "<unknown>");
    }

    if (const auto namedType = mlir::dyn_cast<NamedType>(type); namedType)
    {
      const auto underlyingType = getDITypeAttr(
        context, namedType.getUnderlying(), dataLayout, runtimeTypes, namedType.getName());
      return LLVM::DIDerivedTypeAttr::get(
        context,
        llvm::dwarf::DW_TAG_typedef,
        namedType.getName(),
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
        result = LLVM::DICompositeTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_array_type,
          recId,
          StringAttr::get(context, name),
          diFile,
          0, // LINE
          diScope,
          getDITypeAttr(context, arrayType.getElementType(), dataLayout, runtimeTypes),
          LLVM::DIFlags::Zero,
          size,
          align,
          { diSubrange });
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
        const auto nameAttr = StringAttr::get(context, name);
        result = LLVM::DICompositeTypeAttr::get(
          context,
          llvm::dwarf::DW_TAG_structure_type,
          recId,
          nameAttr,
          diFile,
          0, // LINE
          diScope,
          LLVM::DINullTypeAttr::get(context),
          LLVM::DIFlags::Zero,
          structSizeInBits,
          structAlignmentInBits,
          elementAttrs);
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

  StringRef getArgument() const final { return "go-attach-debug-info-pass"; }

  StringRef getDescription() const final
  {
    return "Attach debug information to local variable allocations";
  }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
};

std::unique_ptr<Pass> createAttachDebugInfoPass()
{
  return std::make_unique<AttachDebugInfoPass>();
}
} // namespace mlir::go
