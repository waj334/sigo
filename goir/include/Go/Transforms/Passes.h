#pragma once

#include <mlir/Pass/Pass.h>

#include <Go/Transforms/TypeConverter.h>

namespace mlir::go
{

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Go/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createExtractTypeMetadataPass();

std::unique_ptr<mlir::Pass> createLowerTypeInfoPass();

std::unique_ptr<mlir::Pass> createDumpToFilePass(StringRef name, StringRef dir);

void populateGoToCoreConversionPatterns(
  mlir::MLIRContext* context,
  TypeConverter& converter,
  RewritePatternSet& patterns);

void populateGoToLLVMConversionPatterns(
  mlir::LLVMTypeConverter& converter,
  RewritePatternSet& patterns);

} // namespace mlir::go
