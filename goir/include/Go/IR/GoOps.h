#ifndef GO_GOOPS_H
#define GO_GOOPS_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/CAPI/IR.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoTypes.h"
#include "Go/Util.h"

#define GET_OP_CLASSES

#include "Go/IR/GoOps.h.inc"

namespace mlir::go {
    template<typename T>
    MlirOperation _createBinOp(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                               MlirLocation location) {
        auto _context = unwrap(context);
        auto _resultType = unwrap(resultType);
        auto _x = unwrap(x);
        auto _y = unwrap(y);
        auto _location = unwrap(location);

        mlir::OpBuilder builder(_context);
        mlir::Operation *op = builder.create<T>(_location, _resultType, _x, _y);
        return wrap(op);
    }

    template<typename T>
    MlirOperation _createUnOp(MlirContext context, MlirValue x, MlirLocation location) {
        auto _context = unwrap(context);
        auto _x = unwrap(x);
        auto _location = unwrap(location);

        mlir::OpBuilder builder(_context);
        mlir::Operation *op = builder.create<T>(_location, _x.getType(), _x);
        return wrap(op);
    }

    static ParseResult
    parseGEPIndices(OpAsmParser &parser,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicIndices,
                    DenseI32ArrayAttr &constIndices) {
        SmallVector<int32_t> constantIndices;
        int32_t index = 0;
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            int32_t constValue;
            OptionalParseResult parsedInteger =
                    parser.parseOptionalInteger(constValue);
            if (parsedInteger.has_value()) {
                if (failed(parsedInteger.value())) {
                    return failure();
                }
                constantIndices.push_back(constValue);
                return success();
            }

            constantIndices.push_back(GetElementPointerOp::kValueFlag | index++);
            return parser.parseOperand(dynamicIndices.emplace_back());
        })) {
            return failure();
        }

        constIndices =
                DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
        return success();
    }

    static void printGEPIndices(OpAsmPrinter &printer, GetElementPointerOp gepOp,
                                OperandRange dynamicIndices,
                                DenseI32ArrayAttr constIndices) {
        llvm::interleaveComma(constIndices.asArrayRef(), printer, [&](int32_t value) {
            if (value & GetElementPointerOp::kValueFlag) {
                const auto index = value & GetElementPointerOp::kValueIndexMask;
                printer.printOperand(dynamicIndices[index]);
            } else {
                printer << value;
            }
        });
    }
}

#endif // GO_GOOPS_H
