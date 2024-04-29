
#pragma once

#include "Go/Transforms/TypeConverter.h"

namespace mlir::go {

    void populateGoToCoreConversionPatterns(mlir::MLIRContext *context,
                                            TypeConverter &converter, RewritePatternSet &patterns);

}
