package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createSlice(ctx context.Context, array llvm.LLVMValueRef, elementType llvm.LLVMTypeRef, numElements uint64, low, high, max llvm.LLVMValueRef) llvm.LLVMValueRef {
	stringType, _ := c.findRuntimeType(ctx, "runtime/internal/go.stringDescriptor")
	sliceType, _ := c.findRuntimeType(ctx, "runtime/internal/go.sliceDescriptor")

	var ptrVal, lengthVal, capacityVal, elementSizeVal llvm.LLVMValueRef

	// Check the type
	arrayType := llvm.TypeOf(array)
	switch arrayType {
	case stringType.valueType:
		ptrVal = llvm.GetAggregateElement(array, 0)
		lengthVal = llvm.GetAggregateElement(array, 1)
		capacityVal = lengthVal
	case sliceType.valueType:
		ptrVal = llvm.GetAggregateElement(array, 0)
		lengthVal = llvm.GetAggregateElement(array, 1)
		capacityVal = llvm.GetAggregateElement(array, 2)
	default:
		if llvm.GetTypeKind(arrayType) == llvm.PointerTypeKind {
			if llvm.IsAAllocaInst(array) != nil {
				arrayType = llvm.GetAllocatedType(array)
				if numElements == 0 {
					kind := llvm.GetTypeKind(arrayType)
					if kind == llvm.ArrayTypeKind {
						numElements = llvm.GetArrayLength2(arrayType)
					} else {
						panic("invalid value type")
					}
				}
				ptrVal = array
				lengthVal = llvm.ConstInt(llvm.Int32Type(), numElements, false)
				capacityVal = lengthVal
			} else {
				panic("invalid value type")
			}
		} else {
			panic("invalid value type")
		}
	}

	// Bitcast the array base pointer to the generic pointer type
	ptrVal = llvm.BuildBitCast(c.builder, ptrVal, c.ptrType.valueType, "")

	if low == nil {
		low = llvm.ConstIntOfString(llvm.Int32Type(), "-1", 10)
	}

	if high == nil {
		high = llvm.ConstIntOfString(llvm.Int32Type(), "-1", 10)
	}

	if max == nil {
		max = llvm.ConstIntOfString(llvm.Int32Type(), "-1", 10)
	}

	elementSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType)
	elementSizeVal = llvm.ConstInt(llvm.Int32Type(), elementSize, false)

	// Create the runtime call
	value, _ := c.createRuntimeCall(ctx, "sliceAddr", []llvm.LLVMValueRef{
		ptrVal, lengthVal, capacityVal, elementSizeVal, low, high, max,
	})

	return value
}

func (c *Compiler) makeInterface(ctx context.Context, value ssa.Value) (result llvm.LLVMValueRef, err error) {
	x, err := c.createExpression(ctx, value)
	if err != nil {
		return nil, err
	}
	typeinfo := c.createTypeDescriptor(ctx, c.createType(ctx, value.Type().Underlying()))

	// Create an alloca for whatever X is so that we can pass the address of it
	alloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(x), "")
	llvm.BuildStore(c.builder, x, alloca)
	xPtr := llvm.BuildBitCast(c.builder, alloca, c.ptrType.valueType, "")

	args := []llvm.LLVMValueRef{xPtr, typeinfo}
	result, err = c.createRuntimeCall(ctx, "makeInterface", args)
	if err != nil {
		return nil, err
	}
	return
}
