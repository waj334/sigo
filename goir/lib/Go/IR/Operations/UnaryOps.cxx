
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

namespace mlir::go {
    void ComplementOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void NegIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    void NotOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }



    ::mlir::ParseResult RecvOp::parse(::mlir::OpAsmParser &p, ::mlir::OperationState &result) {
        ::mlir::OpAsmParser::UnresolvedOperand operand;
        Type resultType;
        mlir::UnitAttr commaOk;

        if (p.parseOperand(operand)) {
            return p.emitError(p.getCurrentLocation(), "operand expected to be `chan` type");
        }

        if (succeeded(p.parseOptionalKeyword("commaOk"))) {
            commaOk = mlir::UnitAttr::get(p.getContext());
            result.addAttribute("commaOk", commaOk);
        }

        if (p.parseColon() || p.parseType(resultType)) {
            return p.emitError(p.getCurrentLocation(), "error parsing result type");
        }

        auto operandType = p.getBuilder().getType<ChanType>(resultType, ChanDirection::SendRecv);
        if (p.resolveOperand(operand, operandType, result.operands)) {
            operandType = p.getBuilder().getType<ChanType>(resultType, ChanDirection::RecvOnly);
            if (p.resolveOperand(operand, operandType, result.operands)) {
                return p.emitError(p.getCurrentLocation(), "could not resolve chan type");
            }
        }

        result.addTypes(resultType);
        return success();
    }

    void RecvOp::print(::mlir::OpAsmPrinter &p) {
        p << " ";
        p.printOperand(this->getOperand());
        if (getCommaOk()) {
            p << " commaOk";
        }
        p << " : ";
        p.printType(this->getType());
    }
}