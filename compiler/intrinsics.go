package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

var intrinsics = map[string]struct{}{
	"AddUint32": {}, "AddInt32": {}, "AddUint64": {}, "AddInt64": {}, "AddUintptr": {},
	"LoadUint32": {}, "LoadInt32": {}, "LoadUint64": {}, "LoadInt64": {}, "LoadUintptr": {}, "LoadPointer": {},
	"StoreUint32": {}, "StoreInt32": {}, "StoreUint64": {}, "StoreInt64": {}, "StoreUintptr": {}, "StorePointer": {},
	"SwapUint32": {}, "SwapInt32": {}, "SwapUint64": {}, "SwapInt64": {}, "SwapUintptr": {}, "SwapPointer": {},
	"CompareAndSwapUint32": {}, "CompareAndSwapInt32": {}, "CompareAndSwapUint64": {}, "CompareAndSwapInt64": {}, "CompareAndSwapUintptr": {}, "CompareAndSwapPointer": {},
}

func (c *Compiler) isIntrinsic(fn *ssa.Function) (ok bool) {
	if fn.Object() != nil {
		name := fn.Object().Id()
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

func (c *Compiler) createIntrinsic(ctx context.Context, call *ssa.Call) (value Value, err error) {
	fn := call.Call.Value.(*ssa.Function)
	name := fn.Object().Id()

	args, err := c.createValues(ctx, call.Call.Args)
	if err != nil {
		return invalidValue, err
	}

	switch name {
	case "AddUint32", "AddInt32", "AddUint64", "AddInt64", "AddUintptr":
		value.LLVMValueRef = llvm.BuildAtomicRMW(c.builder,
			llvm.LLVMAtomicRMWBinOp(llvm.AtomicRMWBinOpAdd),
			args[0],
			args[1],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquireRelease),
			false)
	case "LoadUint32", "LoadInt32", "LoadUint64", "LoadInt64", "LoadUintptr", "LoadPointer":
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, c.createType(ctx, call.Type()).valueType, args[0], "")
		llvm.SetOrdering(value, llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquire))
	case "StoreUint32", "StoreInt32", "StoreUint64", "StoreInt64", "StoreUintptr", "StorePointer":
		str := llvm.BuildStore(c.builder, args[0], args[1])
		llvm.SetOrdering(str, llvm.LLVMAtomicOrdering(llvm.AtomicOrderingRelease))
		value.LLVMValueRef = llvm.GetUndef(llvm.VoidTypeInContext(c.currentContext(ctx)))
	case "SwapUint32", "SwapInt32", "SwapUint64", "SwapInt64", "SwapUintptr", "SwapPointer":
		value.LLVMValueRef = llvm.BuildAtomicRMW(c.builder,
			llvm.LLVMAtomicRMWBinOp(llvm.AtomicRMWBinOpXchg),
			args[0],
			args[1],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingAcquireRelease),
			false)
	case "CompareAndSwapUint32", "CompareAndSwapInt32", "CompareAndSwapUint64", "CompareAndSwapInt64", "CompareAndSwapUintptr", "CompareAndSwapPointer":
		result := llvm.BuildAtomicCmpXchg(c.builder,
			args[0],
			args[1],
			args[2],
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingSequentiallyConsistent),
			llvm.LLVMAtomicOrdering(llvm.AtomicOrderingSequentiallyConsistent),
			false)
		value.LLVMValueRef = llvm.BuildExtractValue(c.builder, result, 1, "")
	default:
		panic("unknown intrinsic " + name)
	}
	return
}
