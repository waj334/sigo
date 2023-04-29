package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createSlice(ctx context.Context, array llvm.LLVMValueRef, elementType llvm.LLVMTypeRef, numElements uint64, low, high, max llvm.LLVMValueRef) llvm.LLVMValueRef {
	stringType := llvm.GetTypeByName2(c.currentContext(ctx), "string")
	sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")

	var ptrVal, lengthVal, capacityVal, elementSizeVal llvm.LLVMValueRef

	// Check the type
	arrayType := llvm.TypeOf(array)
	switch arrayType {
	case stringType:
		ptrVal = llvm.GetAggregateElement(array, 0)
		lengthVal = llvm.GetAggregateElement(array, 1)
		capacityVal = llvm.GetAggregateElement(array, 1)
	case sliceType:
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
						numElements = uint64(llvm.GetArrayLength(arrayType))
					} else {
						panic("invalid value type")
					}
				}
				ptrVal = array
				lengthVal = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), numElements, false)
				capacityVal = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), numElements, false)
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
		low = llvm.ConstIntOfString(llvm.Int32TypeInContext(c.currentContext(ctx)), "-1", 10)
	}

	if high == nil {
		high = llvm.ConstIntOfString(llvm.Int32TypeInContext(c.currentContext(ctx)), "-1", 10)
	}

	if max == nil {
		max = llvm.ConstIntOfString(llvm.Int32TypeInContext(c.currentContext(ctx)), "-1", 10)
	}

	elementSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType)
	elementSizeVal = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), elementSize, false)

	// Create the runtime call
	value, _ := c.createRuntimeCall(ctx, "sliceAddr", []llvm.LLVMValueRef{
		ptrVal, lengthVal, capacityVal, elementSizeVal, low, high, max,
	})

	value = llvm.BuildLoad2(c.builder, sliceType, value, "")

	return value
}

func (c *Compiler) makeInterface(ctx context.Context, value ssa.Value) (result llvm.LLVMValueRef, err error) {
	x, err := c.createExpression(ctx, value)
	if err != nil {
		return nil, err
	}

	interfaceType := llvm.GetTypeByName2(c.currentContext(ctx), "interface")

	// Return the value if it is already of interface type
	if llvm.TypeIsEqual(llvm.TypeOf(x), interfaceType) {
		return x, nil
	}

	typeinfo := c.createTypeDescriptor(ctx, c.createType(ctx, value.Type().Underlying()))
	args := []llvm.LLVMValueRef{c.addressOf(x), typeinfo}
	result, err = c.createRuntimeCall(ctx, "makeInterface", args)
	if err != nil {
		return nil, err
	}

	// Load the interface value
	result = llvm.BuildLoad2(c.builder, interfaceType, result, "")

	return
}

func (c *Compiler) addressOf(value llvm.LLVMValueRef) llvm.LLVMValueRef {
	alloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(value), "")
	llvm.BuildStore(c.builder, value, alloca)
	return llvm.BuildBitCast(c.builder, alloca, c.ptrType.valueType, "")
}

func (c *Compiler) createGlobalString(ctx context.Context, str string) llvm.LLVMValueRef {
	var strArrVal llvm.LLVMValueRef
	if len(str) > 0 {
		strArrVal = llvm.ConstStringInContext(c.currentContext(ctx), str, true)
		strArrVal = c.createGlobalValue(ctx, strArrVal, "global_string_array")
	} else {
		strArrVal = llvm.ConstNull(
			llvm.PointerType(llvm.Int8TypeInContext(c.currentContext(ctx)), 0))
	}

	// Create a string struct
	strType := llvm.GetTypeByName2(c.currentContext(ctx), "string")
	if strType == nil {
		panic("runtime missing stringDescriptor type")
	}

	strVal := llvm.ConstNamedStruct(strType, []llvm.LLVMValueRef{
		strArrVal,
		llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(str)), false),
	})

	return c.createGlobalValue(ctx, strVal, "global_string")
}

func (c *Compiler) createGlobalValue(ctx context.Context, constVal llvm.LLVMValueRef, name string) llvm.LLVMValueRef {
	if !llvm.IsConstant(constVal) {
		panic("attempted to create global from non-const value")
	}

	// Create the global that will hold the constant string value's address
	value := llvm.AddGlobal(c.module, llvm.TypeOf(constVal), name)

	// Set the global variable's value
	llvm.SetInitializer(value, constVal)
	llvm.SetUnnamedAddr(value, llvm.GlobalUnnamedAddr != 0)

	// Bit cast the value
	value = llvm.BuildBitCast(c.builder, value, c.ptrType.valueType, "")

	return value
}
