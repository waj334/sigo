package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

var intrinsics = map[string]struct{}{
	"sync/atomic.AddUint32":  {},
	"sync/atomic.AddInt32":   {},
	"sync/atomic.AddUint64":  {},
	"sync/atomic.AddInt64":   {},
	"sync/atomic.AddUintptr": {},

	"sync/atomic.LoadUint32":  {},
	"sync/atomic.LoadInt32":   {},
	"sync/atomic.LoadUint64":  {},
	"sync/atomic.LoadInt64":   {},
	"sync/atomic.LoadUintptr": {},
	"sync/atomic.LoadPointer": {},

	"sync/atomic.StoreUint32":  {},
	"sync/atomic.StoreInt32":   {},
	"sync/atomic.StoreUint64":  {},
	"sync/atomic.StoreInt64":   {},
	"sync/atomic.StoreUintptr": {},
	"sync/atomic.StorePointer": {},

	"sync/atomic.SwapUint32":  {},
	"sync/atomic.SwapInt32":   {},
	"sync/atomic.SwapUint64":  {},
	"sync/atomic.SwapInt64":   {},
	"sync/atomic.SwapUintptr": {},
	"sync/atomic.SwapPointer": {},

	"sync/atomic.CompareAndSwapUint32":  {},
	"sync/atomic.CompareAndSwapInt32":   {},
	"sync/atomic.CompareAndSwapUint64":  {},
	"sync/atomic.CompareAndSwapInt64":   {},
	"sync/atomic.CompareAndSwapUintptr": {},
	"sync/atomic.CompareAndSwapPointer": {},

	"volatile.LoadInt8":  {},
	"volatile.LoadInt16": {},
	"volatile.LoadInt32": {},
	"volatile.LoadInt64": {},

	"volatile.LoadUint8":   {},
	"volatile.LoadUint16":  {},
	"volatile.LoadUint32":  {},
	"volatile.LoadUint64":  {},
	"volatile.LoadUintptr": {},
	"volatile.LoadPointer": {},

	"volatile.StoreInt8":  {},
	"volatile.StoreInt16": {},
	"volatile.StoreInt32": {},
	"volatile.StoreInt64": {},

	"volatile.StoreUint8":   {},
	"volatile.StoreUint16":  {},
	"volatile.StoreUint32":  {},
	"volatile.StoreUint64":  {},
	"volatile.StoreUintptr": {},
	"volatile.StorePointer": {},
}

func (c *Compiler) isIntrinsic(fn *ssa.Function) (ok bool) {
	if fn.Object() != nil {
		name := c.symbolName(fn.Object().Pkg(), fn.Object().Name())
		_, ok = intrinsics[name]
	}
	return ok
}

func (c *Compiler) isIntrinsicCall(call *ssa.Call) bool {
	fn, ok := call.Call.Value.(*ssa.Function)
	if ok {
		return c.isIntrinsic(fn)
	}
	return false
}

func (c *Compiler) createIntrinsic(ctx context.Context, call *ssa.Call) (value Value) {
	fn := call.Call.Value.(*ssa.Function)
	name := c.symbolName(fn.Object().Pkg(), fn.Name())
	args := c.createValues(ctx, call.Call.Args)

	switch name {
	case "sync/atomic.AddUint32", "sync/atomic.AddInt32", "sync/atomic.AddUint64", "sync/atomic.AddInt64", "sync/atomic.AddUintptr":
		value.LLVMValueRef = llvm.BuildAtomicRMW(c.builder,
			llvm.LLVMAtomicRMWBinOp(llvm.AtomicRMWBinOpAdd),
			args[0],
			args[1],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquireRelease),
			false)
	case "sync/atomic.LoadUint32", "sync/atomic.LoadInt32", "sync/atomic.LoadUint64", "sync/atomic.LoadInt64", "sync/atomic.LoadUintptr", "sync/atomic.LoadPointer":
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, c.createType(ctx, call.Type()).valueType, args[0], "")
		llvm.SetOrdering(value, llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquire))
	case "sync/atomic.StoreUint32", "sync/atomic.StoreInt32", "sync/atomic.StoreUint64", "sync/atomic.StoreInt64", "sync/atomic.StoreUintptr", "sync/atomic.StorePointer":
		str := llvm.BuildStore(c.builder, args[1], args[0])
		llvm.SetOrdering(str, llvm.LLVMAtomicOrdering(llvm.AtomicOrderingRelease))
		value.LLVMValueRef = llvm.GetUndef(llvm.VoidTypeInContext(c.currentContext(ctx)))
	case "sync/atomic.SwapUint32", "sync/atomic.SwapInt32", "sync/atomic.SwapUint64", "sync/atomic.SwapInt64", "sync/atomic.SwapUintptr", "sync/atomic.SwapPointer":
		value.LLVMValueRef = llvm.BuildAtomicRMW(c.builder,
			llvm.LLVMAtomicRMWBinOp(llvm.AtomicRMWBinOpXchg),
			args[0],
			args[1],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquireRelease),
			false)
	case "sync/atomic.CompareAndSwapUint32", "sync/atomic.CompareAndSwapInt32", "sync/atomic.CompareAndSwapUint64", "sync/atomic.CompareAndSwapInt64", "sync/atomic.CompareAndSwapUintptr", "sync/atomic.CompareAndSwapPointer":
		result := llvm.BuildAtomicCmpXchg(c.builder,
			args[0],
			args[1],
			args[2],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingSequentiallyConsistent),
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingSequentiallyConsistent),
			false)
		value.LLVMValueRef = llvm.BuildExtractValue(c.builder, result, 1, "")

	case "volatile.LoadInt8", "volatile.LoadInt16", "volatile.LoadInt32", "volatile.LoadInt64", "volatile.LoadUint8", "volatile.LoadUint16", "volatile.LoadUint32", "volatile.LoadUint64", "volatile.LoadUintptr", "volatile.LoadPointer":
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, c.createType(ctx, call.Type()).valueType, args[0], "")
		llvm.SetVolatile(value, true)
	case "volatile.StoreInt8", "volatile.StoreInt16", "volatile.StoreInt32", "volatile.StoreInt64", "volatile.StoreUint8", "volatile.StoreUint16", "volatile.StoreUint32", "volatile.StoreUint64", "volatile.StoreUintptr", "volatile.StorePointer":
		str := llvm.BuildStore(c.builder, args[1], args[0])
		llvm.SetVolatile(str, true)
		value.LLVMValueRef = llvm.GetUndef(llvm.VoidTypeInContext(c.currentContext(ctx)))
	default:
		panic("unknown intrinsic " + name)
	}
	return
}
