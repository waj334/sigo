#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Interfaces/Utils/InferIntRangeCommon.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    Value convertToInt(PatternRewriter &rewriter, const DataLayout &layout, const Value &input, const Location &loc) {
        Value result = input;
        const Type inputType = input.getType();
        const auto preferredT = IntType::get(rewriter.getContext());

        // Convert the index integer value if it is NOT !go.i.
        if (!go::isa<IntType>(inputType)) {
            if (go::isa<UintType>(inputType) || go::isa<UintptrType>(inputType)) {
                // Bitcast these values to !go.i.
                result = rewriter.create<BitcastOp>(loc, preferredT, input);
            } else {
                const auto indexT = go::cast<IntegerType>(inputType);
                const auto indexBitWidth = layout.getTypeSizeInBits(preferredT);
                if (indexT.getWidth() == indexBitWidth) {
                    result = rewriter.create<BitcastOp>(loc, preferredT, input);
                } else if (indexT.getWidth() > indexBitWidth) {
                    result = rewriter.create<IntTruncateOp>(loc, preferredT, input);
                } else if (indexT.isSigned()) {
                    result = rewriter.create<SignedExtendOp>(loc, preferredT, input);
                } else {
                    result = rewriter.create<ZeroExtendOp>(loc, preferredT, input);
                }
            }
        }
        return result;
    }

    LogicalResult SliceAddrOp::canonicalize(SliceAddrOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        auto elemT = mlir::cast<SliceType>(op.getSlice().getType()).getElementType();
        Value index = op.getIndex();

        // Get the runtime function.
        auto func = mlir::cast<func::FuncOp>(module.lookupSymbol("runtime.sliceIndexAddr"));
        const auto argTypes = func.getArgumentTypes();

        // The info type pointer is the third argument.
        const auto infoType = argTypes[2];

        // Get information about the dynamic type.
        Value info = rewriter.create<TypeInfoOp>(loc, infoType, elemT);

        // Convert the index integer value if it is NOT !go.i.
        const DataLayout layout(module);
        index = convertToInt(rewriter, layout, index, loc);

        // Lower to runtime call.
        const SmallVector<Type> results = {op.getType()};
        const SmallVector<Value> args = {op.getSlice(), index, info};
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, results, "runtime.sliceIndexAddr", args);
        return success();
    }

    void SliceOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
        setResultRange(getResult(), ::mlir::intrange::inferAdd(argRanges));
    }

    LogicalResult SliceOp::verify() {
        // The result can only be a string ONLY if the input is a string.
        if (go::isa<StringType>(this->getType()) && !go::isa<StringType>(this->getInput().getType())) {
            return this->emitOpError() << "the result can only be a string ONLY if the input is a string";
        }
        return success();
    }

    LogicalResult SliceOp::canonicalize(SliceOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        const DataLayout layout(module);
        const auto inputType = op.getInput().getType();
        const auto intT = IntType::get(rewriter.getContext());
        const auto uintptrT = UintptrType::get(rewriter.getContext());

        TypeSwitch<Type>(inputType).Case([&](SliceType T) {
                    // Get the runtime function.
                    auto func = mlir::cast<func::FuncOp>(module.lookupSymbol("runtime.sliceReslice"));
                    const auto argTypes = func.getArgumentTypes();
                    const auto infoType = argTypes[1];

                    // Get information about the slice element type.
                    Value info = rewriter.create<TypeInfoOp>(loc, infoType, T);

                    // Create the runtime call arguments.
                    SmallVector<Value, 5> args = {
                        op.getInput(), info,
                        op.getLow()
                            ? convertToInt(rewriter, layout, op.getLow(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult(),
                        op.getHigh()
                            ? convertToInt(rewriter, layout, op.getHigh(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult(),
                        op.getMax()
                            ? convertToInt(rewriter, layout, op.getMax(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult()
                    };

                    // Replace the operation with the respective runtime call.
                    rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, op->getResults().getTypes(), "runtime.sliceReslice",
                                                               args);
                })
                .Case([&](StringType T) {
                    // Create the runtime call arguments.
                    SmallVector<Value, 3> args = {
                        op.getInput(),
                        op.getLow()
                            ? convertToInt(rewriter, layout, op.getLow(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult(),
                        op.getHigh()
                            ? convertToInt(rewriter, layout, op.getHigh(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult()
                    };

                    // Replace the operation with the respective runtime call.
                    rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, op->getResults().getTypes(), "runtime.stringSlice",
                                                               args);
                })
                .Case([&](PointerType T) {
                    // NOTE: This is a pointer to an array.
                    const auto arrayT = go::cast<ArrayType>(*T.getElementType());
                    const auto elementT = arrayT.getElementType();
                    const auto length = arrayT.getLength();
                    const auto stride = layout.getTypeSize(elementT);

                    // Create the runtime call arguments.
                    SmallVector<Value, 5> args = {
                        op.getInput(),
                        op.getLow()
                            ? convertToInt(rewriter, layout, op.getLow(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult(),
                        op.getHigh()
                            ? convertToInt(rewriter, layout, op.getHigh(), loc)
                            : rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(-1)).getResult(),
                        rewriter.create<ConstantOp>(loc, intT, rewriter.getI32IntegerAttr(length)).getResult(),
                        rewriter.create<ConstantOp>(loc, uintptrT, rewriter.getI32IntegerAttr(stride)).getResult()
                    };

                    // Replace the operation with the respective runtime call.
                    rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, op->getResults().getTypes(), "runtime.sliceAddr",
                                                               args);
                });
        return success();
    }
}
