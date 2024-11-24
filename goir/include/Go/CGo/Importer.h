#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Type.h>

namespace mlir::go
{

class Importer final : public clang::ASTConsumer
{
public:
  explicit Importer(mlir::ModuleOp module);

  mlir::LogicalResult processHeader(const mlir::StringRef& fname);

  void HandleTranslationUnit(clang::ASTContext& context) override;

  mlir::Type translateType(clang::ASTContext& context, clang::QualType type);

private:
  mlir::ModuleOp m_module;
  mlir::MLIRContext* mp_context;
  mlir::DenseMap<const clang::Type*, mlir::Type> m_typeCache;
};

} // namespace mlir::go
