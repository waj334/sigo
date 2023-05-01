package compiler

import (
	"context"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

type Value struct {
	llvm.LLVMValueRef
	dbg  llvm.LLVMMetadataRef
	heap bool
	cc   *Compiler
}

func (v Value) Kind() llvm.LLVMTypeKind {
	return llvm.GetTypeKind(llvm.TypeOf(v))
}

func (v Value) Linkage() llvm.LLVMLinkage {
	return llvm.GetLinkage(v)
}

func (v Value) UnderlyingValue() llvm.LLVMValueRef {
	ref := v.LLVMValueRef
	if llvm.IsAAllocaInst(ref) != nil && v.heap {
		// Load the ptr from the alloca
		ref = llvm.BuildLoad2(v.cc.builder, v.cc.ptrType.valueType, ref, "")
	}
	return ref
}

type Values []Value

func (v Values) Ref() []llvm.LLVMValueRef {
	refs := make([]llvm.LLVMValueRef, 0, len(v))
	for _, val := range v {
		refs = append(refs, val)
	}
	return refs
}

func (c *Compiler) createValues(ctx context.Context, input []ssa.Value) (Values, error) {
	var output []Value
	for _, in := range input {
		// Evaluate the argument
		value, err := c.createExpression(ctx, in)
		if err != nil {
			return nil, err
		}

		// Append to the args list that will be passed to the function
		output = append(output, value)
	}
	return output, nil
}

func (c *Compiler) createVariable(ctx context.Context, name string, value Value, valueType types.Type) Value {
	c.printf(Debug, "Creating variable for %s (%s)\n", name, valueType.String())
	defer c.printf(Debug, "Done creating variable for %s (%s)\n", name, valueType.String())

	// Create and associate the variable with the value
	fn := c.currentFunction(ctx)
	dbgType := c.createDebugType(ctx, valueType)

	// Create the debug information about the variable
	value.dbg = llvm.DIBuilderCreateAutoVariable(
		c.dibuilder,
		fn.subprogram,
		name,
		fn.diFile,
		uint(c.currentLocation(ctx).position.Line),
		dbgType,
		true,
		0,
		0)

	// Add debug info about the declaration
	llvm.DIBuilderInsertDeclareAtEnd(
		c.dibuilder,
		value,
		value.dbg,
		llvm.DIBuilderCreateExpression(c.dibuilder, nil),
		c.currentLocation(ctx).line,
		c.currentBlock(ctx))

	return value
}

func (c *Compiler) createSlice(ctx context.Context, array llvm.LLVMValueRef, elementType llvm.LLVMTypeRef, numElements uint64, low, high, max llvm.LLVMValueRef) llvm.LLVMValueRef {
	stringType := llvm.GetTypeByName2(c.currentContext(ctx), "string")
	sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")

	var ptrVal, lengthVal, capacityVal, elementSizeVal llvm.LLVMValueRef

	// Check the type
	arrayType := llvm.TypeOf(array)
	isString := false
	switch arrayType {
	case stringType:
		ptrVal = llvm.BuildExtractValue(c.builder, array, 0, "")
		lengthVal = llvm.BuildExtractValue(c.builder, array, 1, "")
		capacityVal = lengthVal
		isString = true
	case sliceType:
		ptrVal = llvm.BuildExtractValue(c.builder, array, 0, "")
		lengthVal = llvm.BuildExtractValue(c.builder, array, 1, "")
		capacityVal = llvm.BuildExtractValue(c.builder, array, 2, "")
	default:
		if llvm.GetTypeKind(arrayType) == llvm.PointerTypeKind {
			ptrVal = array
			lengthVal = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), numElements, false)
			capacityVal = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), numElements, false)
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

	// Return a new string if the input was a string
	if isString {
		ptrVal = llvm.BuildExtractValue(c.builder, value, 0, "")
		lengthVal = llvm.BuildExtractValue(c.builder, value, 1, "")

		// Create a new string
		value = llvm.BuildAlloca(c.builder, stringType, "")
		arrayAddr := llvm.BuildStructGEP2(c.builder, stringType, value, 0, "")
		lenAddr := llvm.BuildStructGEP2(c.builder, stringType, value, 1, "")
		llvm.BuildStore(c.builder, ptrVal, arrayAddr)
		llvm.BuildStore(c.builder, lengthVal, lenAddr)

		// Load the string value
		value = llvm.BuildLoad2(c.builder, stringType, value, "")
	}

	return value
}

func (c *Compiler) createSliceFromValues(ctx context.Context, values []llvm.LLVMValueRef) llvm.LLVMValueRef {
	// Check all values to make sure they are the same type
	var lastType llvm.LLVMTypeRef
	for _, value := range values {
		if lastType == nil {
			lastType = llvm.TypeOf(value)
			continue
		} else if !llvm.TypeIsEqual(lastType, llvm.TypeOf(value)) {
			panic("slices values are different types")
		}
	}

	// Create the underlying array for the slice
	constLen := llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(values)), false)
	array := llvm.BuildArrayAlloca(c.builder, lastType, constLen, "")

	// Populate the underlying array for the slice
	for i, value := range values {
		index := llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(i), false)
		addr := llvm.BuildGEP2(c.builder, lastType, array, []llvm.LLVMValueRef{index}, "")
		llvm.BuildStore(c.builder, value, addr)
	}

	// Create the slice struct
	sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")
	result := llvm.BuildAlloca(c.builder, sliceType, "")

	arrayAddr := llvm.BuildStructGEP2(c.builder, sliceType, result, 0, "")
	lenAddr := llvm.BuildStructGEP2(c.builder, sliceType, result, 1, "")
	capAddr := llvm.BuildStructGEP2(c.builder, sliceType, result, 2, "")

	llvm.BuildStore(c.builder, array, arrayAddr)
	llvm.BuildStore(c.builder, constLen, lenAddr)
	llvm.BuildStore(c.builder, constLen, capAddr)

	// Finally, return the slice
	return result
}

func (c *Compiler) createSliceFromStringValue(ctx context.Context, str llvm.LLVMValueRef) llvm.LLVMValueRef {
	sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")
	stringType := llvm.GetTypeByName2(c.currentContext(ctx), "string")

	if !llvm.TypeIsEqual(stringType, llvm.TypeOf(str)) {
		panic("value is not a string")
	}

	// Create the slice struct
	slice := llvm.BuildAlloca(c.builder, sliceType, "")

	arrayAddr := llvm.BuildStructGEP2(c.builder, sliceType, slice, 0, "")
	lenAddr := llvm.BuildStructGEP2(c.builder, sliceType, slice, 1, "")
	capAddr := llvm.BuildStructGEP2(c.builder, sliceType, slice, 2, "")

	arrayValue := llvm.BuildExtractValue(c.builder, str, 0, "")
	lengthValue := llvm.BuildExtractValue(c.builder, str, 1, "")

	llvm.BuildStore(c.builder, arrayValue, arrayAddr)
	llvm.BuildStore(c.builder, lengthValue, lenAddr)
	llvm.BuildStore(c.builder, lengthValue, capAddr)

	return slice
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

func (c *Compiler) createConstantString(ctx context.Context, str string) llvm.LLVMValueRef {
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
		panic("missing string type")
	}

	return llvm.ConstNamedStruct(strType, []llvm.LLVMValueRef{
		strArrVal,
		llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(str)), false),
	})
}

func (c *Compiler) createGlobalString(ctx context.Context, str string) llvm.LLVMValueRef {
	strVal := c.createConstantString(ctx, str)
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

func (c *Compiler) structFieldAddress(value Value, structType llvm.LLVMTypeRef, index int) (llvm.LLVMTypeRef, llvm.LLVMValueRef) {
	// Get the type of the field
	fieldType := llvm.StructGetTypeAtIndex(structType, uint(index))

	// Create a GEP to get the  address of the field in the struct
	return fieldType, llvm.BuildStructGEP2(c.builder, structType, value.UnderlyingValue(), uint(index), "")
}
