
#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include "Go/IR/GoOps.h"

namespace mlir::go
{

    void populateGoToLLVMConversionPatterns(
            mlir::LLVMTypeConverter &converter, RewritePatternSet &patterns);

} // namespace mlir::go