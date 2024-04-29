#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
        // Constant op constant-folds to its value.
        return getValue();
    }
}