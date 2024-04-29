#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Pass/Pass.h>

#include "Go/Transforms/TypeConverter.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    void populateTypeInformationToGoConversionPatterns(mlir::ModuleOp module, mlir::go::LLVMTypeConverter&converter,
                                                       RewritePatternSet&patterns);

    std::unique_ptr<mlir::Pass> createLowerTypeInfoPass();
} // namespace mlir::go
