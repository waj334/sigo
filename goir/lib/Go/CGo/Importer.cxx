
#include "Go/CGo/Importer.h"

#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/Builders.h>

#include <Go/IR/GoOps.h.inc>
#include <Go/IR/GoOps.h>
#include <Go/IR/GoTypes.h.inc>
#include <Go/IR/GoTypes.h>
#include <clang/AST/AST.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Parse/ParseAST.h>

namespace mlir::go
{

Importer::Importer(mlir::ModuleOp module)
  : m_module(module)
  , mp_context(m_module.getContext())
{
}

mlir::LogicalResult Importer::processHeader(const mlir::StringRef& fname)
{
  clang::CompilerInstance compilerInstance;
  compilerInstance.createDiagnostics();

  // Get the target triple from the module.
  const auto targetTriple =
    this->m_module->getAttrOfType<mlir::StringAttr>("llvm.target_triple").getValue();

  // Configure the target options.
  const auto targetOptions = std::make_shared<clang::TargetOptions>();
  targetOptions->Triple = targetTriple;

  const auto targetInfo =
    clang::TargetInfo::CreateTargetInfo(compilerInstance.getDiagnostics(), targetOptions);
  compilerInstance.setTarget(targetInfo);

  // Create managers.
  compilerInstance.createFileManager();
  compilerInstance.createSourceManager(compilerInstance.getFileManager());

  // Set the main file to parse.
  const auto file = compilerInstance.getFileManager().getFileRef(fname).get();
  const auto fid = compilerInstance.getSourceManager().createFileID(
    file, clang::SourceLocation(), clang::SrcMgr::C_User);
  compilerInstance.getSourceManager().setMainFileID(fid);

  // Set up the preprocessor.
  compilerInstance.createPreprocessor(clang::TranslationUnitKind::TU_Complete);
  compilerInstance.getPreprocessorOpts().UsePredefines = false;

  // Process the source file.
  compilerInstance.createASTContext();
  clang::ParseAST(compilerInstance.getPreprocessor(), this, compilerInstance.getASTContext());

  return mlir::success();
}
void Importer::HandleTranslationUnit(clang::ASTContext& context)
{
  mlir::OpBuilder builder(this->m_module->getRegion(0));
  mlir::Location loc = mlir::UnknownLoc::get(this->m_module->getContext());

  const clang::TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  for (const clang::Decl* _decl : tu->decls())
  {
    llvm::TypeSwitch<const clang::Decl*>(_decl)
      .Case(
        [&](const clang::FunctionDecl* decl)
        {
          if (decl->isGlobal())
          {
            // Emit a function declaration into the module.
            const auto symbolName = decl->getName();
            // builder.create<mlir::go::FuncOp>(loc, symbolName);
          }
        })
      .Case(
        [&](const clang::VarDecl* decl)
        {
          // TODO
        })
      .Case(
        [&](const clang::TypeDecl* decl)
        {
          // TODO
        });
  }
}
mlir::Type Importer::translateType(clang::ASTContext& context, clang::QualType _type)
{
  mlir::Type result;
  if (const auto it = this->m_typeCache.find(_type.getTypePtr()); it != this->m_typeCache.end())
  {
    return it->second;
  }

  const size_t width = context.getTypeSize(_type);
  llvm::TypeSwitch<const clang::Type*>(_type.getTypePtr())
    .Case(
      [&](const clang::ConstantArrayType* type)
      {
        const auto elementType = this->translateType(context, type->getElementType());
        result = mlir::go::ArrayType::get(this->mp_context, elementType, type->getZExtSize());
      })
    .Case(
      [&](const clang::BuiltinType* type)
      {
        if (type->isSignedInteger())
        {
          result =
            mlir::go::IntegerType::get(this->mp_context, mlir::go::IntegerType::Signed, width);
        }
        else if (type->isUnsignedInteger())
        {
          result =
            mlir::go::IntegerType::get(this->mp_context, mlir::go::IntegerType::Unsigned, width);
        }
        else if (type->isFloatingPoint())
        {
          if (width == 32)
          {
            result = mlir::FloatType::getF32(this->mp_context);
          }
          else
          {
            result = mlir::FloatType::getF64(this->mp_context);
          }
        }
        else
        {
          llvm_unreachable("unhandled builtin type");
        }
      })
    .Case(
      [&](const clang::ComplexType* type)
      {
        if (width == 64)
        {
          result = mlir::ComplexType::get(mlir::FloatType::getF32(this->mp_context));
        }
        else
        {
          result = mlir::ComplexType::get(mlir::FloatType::getF64(this->mp_context));
        }
      })
    .Case(
      [&](const clang::PointerType* type)
      {
        const auto elementType = this->translateType(context, type->getPointeeType());
        result = mlir::go::PointerType::get(this->mp_context, elementType);
      })
    .Default([&](auto) { llvm_unreachable("unhandled type"); });

  this->m_typeCache[_type.getTypePtr()] = result;
  return result;
}

} // namespace mlir::go