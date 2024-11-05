#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    LogicalResult MakeChanOp::canonicalize(MakeChanOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        const auto resultType = mlir::cast<ChanType>(op.getType());
        const auto calleeSymbol = formatPackageSymbol("runtime", "channelMake");

        // Get the runtime function.
        auto func = module.lookupSymbol<FuncOp>(calleeSymbol);
        const auto argTypes = func.getArgumentTypes();

        // The info type pointer is the first argument.
        const auto infoType = argTypes[0];

        // Get information about the element type.
        Value info = rewriter.create<TypeInfoOp>(loc, infoType, resultType.getElementType());

        // Lower to runtime call.
        const SmallVector<Type> results = {op.getType()};
        const SmallVector<Value> args = {info, op.getCapacity()};
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, results, calleeSymbol, args);
        return success();
    }

    LogicalResult MakeMapOp::canonicalize(MakeMapOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        const auto resultType = mlir::cast<MapType>(op.getType());
        const auto calleeSymbol = formatPackageSymbol("runtime", "mapMake");

        // Get the runtime function.
        auto func = module.lookupSymbol<FuncOp>(calleeSymbol);
        const auto argTypes = func.getArgumentTypes();

        // The info type pointer is the first argument.
        const auto infoType = argTypes[0];

        // Get information about the key and element type.
        Value keyInfo = rewriter.create<TypeInfoOp>(loc, infoType, resultType.getKeyType());
        Value elementInfo = rewriter.create<TypeInfoOp>(loc, infoType, resultType.getValueType());

        // Lower to runtime call.
        const SmallVector<Type> results = {op.getType()};
        const SmallVector<Value> args = {keyInfo, elementInfo, op.getCapacity()};
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, results, calleeSymbol, args);
        return success();
    }

    LogicalResult MakeSliceOp::canonicalize(MakeSliceOp op, PatternRewriter &rewriter) {
        const auto loc = op.getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        const auto resultType = mlir::cast<SliceType>(op.getType());
        const auto calleeSymbol = formatPackageSymbol("runtime", "sliceMake");

        // Get the runtime function.
        auto func = module.lookupSymbol<FuncOp>(calleeSymbol);
        const auto argTypes = func.getArgumentTypes();

        // The info type pointer is the first argument.
        const auto infoType = argTypes[0];

        // Get information about the element type.
        Value info = rewriter.create<TypeInfoOp>(loc, infoType, resultType.getElementType());

        // Lower to runtime call.
        const SmallVector<Type, 1> results = {op.getType()};
        const SmallVector<Value, 3> args = {info, op.getLength(), op.getCapacity() ? op.getCapacity() : op.getLength()};

        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, results, calleeSymbol, args);
        return success();
    }

}
