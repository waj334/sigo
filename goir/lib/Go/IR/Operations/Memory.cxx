#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

namespace mlir::go {
    ::mlir::LogicalResult AllocaOp::verify() {
        auto resultT = go::dyn_cast<PointerType>(this->getType());
        if (!resultT) {
            return this->emitOpError() << "the alloca operation must return a pointer type";
        }

        if (resultT.getElementType() && *resultT.getElementType() != this->getElement()) {
            return this->emitOpError() << "the alloca operation must return either !go.ptr or !go.ptr<" << this->
                   getElement()
                   << ">";
        }

        return success();
    }

    ::mlir::LogicalResult StoreOp::verify() {
        auto addrType = go::cast<PointerType>(this->getAddr().getType());
        if (!addrType) {
            return this->emitOpError() << "address type must be a pointer";
        }

        if (addrType.getElementType().has_value() && *addrType.getElementType() != this->getValue().getType()) {
            return this->emitOpError() << "value type " << this->getValue().getType() <<
                   " is incompatible with pointer type "
                   << addrType;
        }

        return success();
    }

    ::mlir::LogicalResult GetElementPointerOp::verify() {
        // No constant index should be negative.
        for (auto value: this->getConstIndices()) {
            if (value < 0 && (value & kValueIndexMask) > this->getDynamicIndices().size()) {
                return this->emitOpError() << "constant indices cannot be negative";
            }
        }
        return success();
    }

    ::mlir::LogicalResult GlobalOp::verify() {
        // Globals MUST specify a type.
        if (!this->getGlobalType()) {
            return this->emitOpError() << "globals MUST specify a type";
        }
        return success();
    }

    ::mlir::LogicalResult YieldOp::verify() {
        auto globalOp = this->getParentOp<GlobalOp>();
        if (!globalOp)
        {
          return this->emitOpError() << "yield op must have global op as a parent";
        }

        const auto expectedType = globalOp.getGlobalType();
        const auto actualType = this->getInitializerValue().getType();
        if (actualType != expectedType) {
            return this->emitOpError() << "expected to yield value: " << expectedType << "\ngot:" << actualType;
        }
        return success();
    }
} // namespace mlir::go
