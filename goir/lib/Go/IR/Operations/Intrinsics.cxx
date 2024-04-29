#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    /*LogicalResult TypeInfoOp::verify() {
        auto resultType = this->getResult().getType().dyn_cast<PointerType>();
        // Resulting pointer must NOT specify a type
        if (resultType.getElementType().has_value()) {
            return failure();
        }
        return success();
    }*/
}