
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

namespace mlir::go {
    void AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void AndNotOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void ShrUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void ShrSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    ::mlir::LogicalResult CmpCOp::verify() {
        // Only == and != operators allowed for complex numbers as they are not ordered.
        if (this->getPredicate() != CmpFPredicate::eq && this->getPredicate() != CmpFPredicate::ne)
        {
            return emitOpError() << "only `==` and `!=` operators allowed for complex numbers as they are not ordered";
        }
        return success();
    }
}