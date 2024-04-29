package ssa

import (
	"context"
	"go/ast"
	"go/types"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) isIntrinsic(expr *ast.CallExpr) bool {
	if obj, ok := b.objectOf(expr.Fun).(*types.Func); ok {
		return isIntrinsic(qualifiedFuncName(obj))
	}
	return false
}

func isIntrinsic(symbol string) bool {
	switch symbol {
	case "sync/atomic.AddUint32",
		"sync/atomic.AddInt32",
		"sync/atomic.AddUint64",
		"sync/atomic.AddInt64",
		"sync/atomic.AddUintptr",

		"sync/atomic.LoadUint32",
		"sync/atomic.LoadInt32",
		"sync/atomic.LoadUint64",
		"sync/atomic.LoadInt64",
		"sync/atomic.LoadUintptr",
		"sync/atomic.LoadPointer",

		"sync/atomic.StoreUint32",
		"sync/atomic.StoreInt32",
		"sync/atomic.StoreUint64",
		"sync/atomic.StoreInt64",
		"sync/atomic.StoreUintptr",
		"sync/atomic.StorePointer",

		"sync/atomic.SwapUint32",
		"sync/atomic.SwapInt32",
		"sync/atomic.SwapUint64",
		"sync/atomic.SwapInt64",
		"sync/atomic.SwapUintptr",
		"sync/atomic.SwapPointer",

		"sync/atomic.CompareAndSwapUint32",
		"sync/atomic.CompareAndSwapInt32",
		"sync/atomic.CompareAndSwapUint64",
		"sync/atomic.CompareAndSwapInt64",
		"sync/atomic.CompareAndSwapUintptr",
		"sync/atomic.CompareAndSwapPointer",

		"volatile.LoadInt8",
		"volatile.LoadInt16",
		"volatile.LoadInt32",
		"volatile.LoadInt64",
		"volatile.LoadUint8",
		"volatile.LoadUint16",
		"volatile.LoadUint32",
		"volatile.LoadUint64",
		"volatile.LoadUintptr",
		"volatile.LoadPointer",

		"volatile.StoreInt8",
		"volatile.StoreInt16",
		"volatile.StoreInt32",
		"volatile.StoreInt64",
		"volatile.StoreUint8",
		"volatile.StoreUint16",
		"volatile.StoreUint32",
		"volatile.StoreUint64",
		"volatile.StoreUintptr",
		"volatile.StorePointer":
		return true
	default:
		return false
	}
}

func (b *Builder) emitIntrinsic(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	F := b.objectOf(expr.Fun).(*types.Func)
	signature := F.Type().Underlying().(*types.Signature)
	symbol := qualifiedFuncName(F)

	location := b.location(expr.Pos())
	args := make([]mlir.Value, len(expr.Args))
	for i, expr := range expr.Args {
		ctx = newContextWithRhsIndex(ctx, i)
		args[i] = b.emitExpr(ctx, expr)[0]
	}

	switch symbol {
	case "sync/atomic.AddUint32", "sync/atomic.AddInt32", "sync/atomic.AddUint64", "sync/atomic.AddInt64", "sync/atomic.AddUintptr":
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := mlir.GoCreateAtomicAddIOperation(b.ctx, T, args[0], args[1], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.CompareAndSwapUint32", "sync/atomic.CompareAndSwapInt32", "sync/atomic.CompareAndSwapUint64", "sync/atomic.CompareAndSwapInt64", "sync/atomic.CompareAndSwapUintptr", "sync/atomic.CompareAndSwapPointer":
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := mlir.GoCreateAtomicCompareAndSwapOperation(b.ctx, T, args[0], args[1], args[2], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.LoadUint32", "sync/atomic.LoadInt32", "sync/atomic.LoadUint64", "sync/atomic.LoadInt64", "sync/atomic.LoadUintptr", "sync/atomic.LoadPointer":
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := mlir.GoCreateAtomicLoadOperation(b.ctx, args[0], T, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.StoreUint32", "sync/atomic.StoreInt32", "sync/atomic.StoreUint64", "sync/atomic.StoreInt64", "sync/atomic.StoreUintptr", "sync/atomic.StorePointer":
		op := mlir.GoCreateAtomicStoreOperation(b.ctx, args[1], args[0], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.SwapUint32", "sync/atomic.SwapInt32", "sync/atomic.SwapUint64", "sync/atomic.SwapInt64", "sync/atomic.SwapUintptr", "sync/atomic.SwapPointer":
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := mlir.GoCreateAtomicSwapOperation(b.ctx, T, args[0], args[1], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "volatile.LoadInt8", "volatile.LoadInt16", "volatile.LoadInt32", "volatile.LoadInt64", "volatile.LoadUint8", "volatile.LoadUint16", "volatile.LoadUint32", "volatile.LoadUint64", "volatile.LoadUintptr", "volatile.LoadPointer":
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := mlir.GoCreateVolatileLoadOperation(b.ctx, args[0], T, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "volatile.StoreInt8", "volatile.StoreInt16", "volatile.StoreInt32", "volatile.StoreInt64", "volatile.StoreUint8", "volatile.StoreUint16", "volatile.StoreUint32", "volatile.StoreUint64", "volatile.StoreUintptr", "volatile.StorePointer":
		op := mlir.GoCreateVolatileStoreOperation(b.ctx, args[1], args[0], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		panic("unhandled")
	}
}
