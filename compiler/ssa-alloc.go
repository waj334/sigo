package compiler

import (
	"context"
	"go/types"
	"omibyte.io/sigo/llvm"
)

type Alloc struct {
	expressionBase
	Heap    bool
	Comment string
}

func (a Alloc) Generate(ctx context.Context) (result Value) {
	elementType := a.cc.createType(ctx, a.goType.Underlying().(*types.Pointer).Elem())

	// NOTE: Some stack allocations will be moved to the heap later if they
	//       are too big for the stack.
	// Get the size of the type
	size := llvm.StoreSizeOfType(a.cc.options.Target.dataLayout, elementType.valueType)

	if a.Heap {
		// Create the alloca to hold the address on the stack
		result.LLVMValueRef = a.cc.createAlloca(ctx, a.cc.ptrType.valueType, a.Comment)

		// Mark this value as one that is on the heap
		result.heap = true

		// Next, create the debug info. This heap allocation will be treated differently
		result = a.cc.createVariable(ctx, a.Comment, result, a.goType.Underlying())

		// Create the runtime call to allocate some memory on the heap
		obj := a.cc.createRuntimeCall(ctx, "alloc",
			[]llvm.LLVMValueRef{llvm.ConstInt(a.cc.uintptrType.valueType, size, false)})

		// Store the address at the alloc
		llvm.BuildStore(a.cc.builder, obj, result.LLVMValueRef)
	} else {
		// Create an alloca to hold the value on the stack
		result.LLVMValueRef = a.cc.createAlloca(ctx, elementType.valueType, a.Comment)
		result = a.cc.createVariable(ctx, a.Comment, result, elementType.spec)

		result.heap = false

		// Zero-initialize the stack variable
		if size > 0 {
			llvm.BuildStore(a.cc.builder, llvm.ConstNull(elementType.valueType), result.LLVMValueRef)
		}
	}
	return
}
