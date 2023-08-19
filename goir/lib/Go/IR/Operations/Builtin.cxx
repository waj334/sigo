#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "Go/Util.h"

namespace mlir::go {

    /*mlir::SuccessorOperands PanicOp::getSuccessorOperands(unsigned index) {
        assert(index < getNumSuccessors() && "invalid successor index");
        return mlir::SuccessorOperands(this->getSuccessor());
    }*/

}
