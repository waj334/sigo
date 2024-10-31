#pragma once

#include <mlir/Pass/Pass.h>

#include <Go/Transforms/TypeConverter.h>

namespace mlir::go
{
std::unique_ptr<Pass> createAttachDebugInfoPass();

std::unique_ptr<Pass> createExtractTypeMetadataPass();

std::unique_ptr<mlir::Pass> createGlobalConstantsPass();

std::unique_ptr<Pass> createGlobalInitializerPass();

std::unique_ptr<mlir::Pass> createHeapEscapePass();

std::unique_ptr<mlir::Pass> createLowerToCorePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createLowerTypeInfoPass();

std::unique_ptr<mlir::Pass> createDumpToFilePass(StringRef name, StringRef dir);

std::unique_ptr<mlir::Pass> createOptimizeDefersPass();

std::unique_ptr<mlir::Pass> createCallPass();

void populateGoToCoreConversionPatterns(
  mlir::MLIRContext* context,
  TypeConverter& converter,
  RewritePatternSet& patterns);

void populateGoToLLVMConversionPatterns(
  mlir::LLVMTypeConverter& converter,
  RewritePatternSet& patterns);

} // namespace mlir::go
