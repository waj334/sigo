#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir::go {
    LogicalResult ExtractOp::verify() {
        uint64_t numElements = -1;
        SmallVector<Type> elements;
        auto result = ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(baseType(this->getAggregate().getType()))
                .Case([&](GoStructType T) {
                    elements = T.getFieldTypes();
                    numElements = elements.size();
                    return success();
                })
                .Case([&](ArrayType T) {
                    numElements = T.getLength();
                    elements = SmallVector<Type>(numElements, T.getElementType());
                    return success();
                })
                .Default([&](Type T) { return this->emitOpError() << "unexpected aggregate type " << T; });

        if (failed(result))
            return result;

        // Assert that the index is in range
        if (this->getIndex() >= numElements) {
            return this->emitOpError() << "index out of range";
        }

        if (!elements.empty()) {
            auto elemT = elements[this->getIndex()];
            auto resultT = this->getResult().getType();

            // Assert that the type being extracted matches that of the respective field in the struct
            if (elemT != resultT) {
                return this->emitOpError() << "element type at index " << this->getIndex() <<
                       " (" << elemT << ") does not match result type (" << resultT << ")";
            }
        }
        return success();
    }

    LogicalResult InsertOp::verify() {
        uint64_t numElements = -1;
        SmallVector<Type> elements;
        auto result = ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(baseType(this->getAggregate().getType()))
                .Case([&](GoStructType T) {
                    elements = T.getFieldTypes();
                    numElements = elements.size();
                    return success();
                })
                .Case([&](ArrayType T) {
                    numElements = T.getLength();
                    elements = SmallVector<Type>(numElements, T.getElementType());
                    return success();
                })
                .Default([&](Type T) { return this->emitOpError() << "unexpected aggregate type " << T; });

        if (failed(result))
            return result;

        // Assert that the index is in range
        if (this->getIndex() >= numElements) {
            return this->emitOpError() << "index out of range";
        }

        if (!elements.empty()) {
            auto elemT = elements[this->getIndex()];
            auto inputT = this->getValue().getType();

            // Assert that the type being extracted matches that of the respective field in the struct
            if (elemT != inputT) {
                return this->emitOpError() << "element type at index " << this->getIndex() <<
                       " (" << elemT << ") does not match input value type (" << inputT << ")";
            }
        }
        return success();
    }
} // namespace mlir::go
