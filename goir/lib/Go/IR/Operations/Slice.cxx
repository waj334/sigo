#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    LogicalResult SliceAddrOp::canonicalize(SliceAddrOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        auto elemT = mlir::cast<SliceType>(op.getSlice().getType()).getElementType();
        auto _indexT = op.getIndex().getType();
        Value index = op.getIndex();
        const auto preferredIndexT = IntType::get(rewriter.getContext());

        // Get the runtime function.
        auto func = mlir::cast<func::FuncOp>(module.lookupSymbol("runtime.sliceIndexAddr"));
        const auto argTypes = func.getArgumentTypes();

        // The info type pointer is the third argument.
        const auto infoType = argTypes[2];

        // Get information about the dynamic type.
        Value info = rewriter.create<TypeInfoOp>(loc, infoType, elemT);

        // Convert the index integer value if it is NOT !go.i.
        if (!go::isa<IntType>(_indexT)) {
            if (go::isa<UintType>(_indexT) || go::isa<UintptrType>(_indexT)) {
                // Bitcast these values to !go.i.
                index = rewriter.create<BitcastOp>(loc, preferredIndexT, index);
            } else {
                DataLayout layout(module);
                const auto indexT = go::cast<IntegerType>(_indexT);
                const auto indexBitWidth = layout.getTypeSizeInBits(preferredIndexT);
                if (indexT.getWidth() == indexBitWidth) {
                    index = rewriter.create<BitcastOp>(loc, preferredIndexT, index);
                } else if (indexT.getWidth() > indexBitWidth) {
                    index = rewriter.create<IntTruncateOp>(loc, preferredIndexT, index);
                } else if (indexT.isSigned()) {
                    index = rewriter.create<SignedExtendOp>(loc, preferredIndexT, index);
                } else {
                    index = rewriter.create<ZeroExtendOp>(loc, preferredIndexT, index);
                }
            }
        }

        // Lower to runtime call.
        const SmallVector<Type> results = {op.getType()};
        const SmallVector<Value> args = {op.getSlice(), index, info};
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, results, "runtime.sliceIndexAddr", args);
        return success();
    }
}
