package compiler

import (
	"context"
	"go/token"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
	"path/filepath"
	"strings"
)

type Value struct {
	ref      llvm.LLVMValueRef
	dbg      llvm.LLVMMetadataRef
	heap     bool
	cc       *Compiler
	extern   bool
	exported bool
	global   bool
	linkname string
	spec     ssa.Value
}

func (v Value) Kind() llvm.LLVMTypeKind {
	return llvm.GetTypeKind(llvm.TypeOf(v.ref))
}

func (v Value) Linkage() llvm.LLVMLinkage {
	return llvm.GetLinkage(v.ref)
}

func (v Value) UnderlyingValue(ctx context.Context) llvm.LLVMValueRef {
	ref := v.ref
	if llvm.IsAAllocaInst(ref) != nil && v.heap {
		typ := v.cc.createType(ctx, v.spec.Type())

		// Heap allocations are ptr->ptr. Load the address of the actual value
		ref = llvm.BuildLoad2(v.cc.builder, typ.valueType, ref, "heap_data_start")
	}
	return ref
}

func (v Value) File() *token.File {
	var pkg *ssa.Package
	switch value := v.spec.(type) {
	case *ssa.Global:
		pkg = value.Package()
	default:
		pkg = value.Parent().Pkg
	}
	return pkg.Prog.Fset.File(v.spec.Pos())
}

func (v Value) DebugFile() llvm.LLVMMetadataRef {
	filename := "<unknown>"

	file := v.File()
	if file != nil {
		// Extract the file info
		filename = v.cc.options.MapPath(v.File().Name())
	}

	// Return the file
	return llvm.DIBuilderCreateFile(
		v.cc.dibuilder,
		filepath.Base(filename),
		filepath.Dir(filename))
}

func (v Value) Pos() token.Position {
	return v.File().Position(v.spec.Pos())
}

func (v Value) DebugPos(ctx context.Context) llvm.LLVMMetadataRef {
	return llvm.DIBuilderCreateDebugLocation(
		v.cc.currentContext(ctx),
		uint(v.Pos().Line),
		uint(v.Pos().Line),
		llvm.GetSubprogram(llvm.GetBasicBlockParent(llvm.GetInstructionParent(v.ref))),
		nil,
	)
}

type Values []Value

func (v Values) Ref(ctx context.Context) []llvm.LLVMValueRef {
	refs := make([]llvm.LLVMValueRef, 0, len(v))
	for _, val := range v {
		refs = append(refs, val.UnderlyingValue(ctx))
	}
	return refs
}

func (c *Compiler) createValues(ctx context.Context, input []ssa.Value) Values {
	var output []Value
	for _, in := range input {
		// Evaluate the argument
		value := c.createExpression(ctx, in)

		// Append to the args list that will be passed to the function
		output = append(output, value)
	}
	return output
}

func (c *Compiler) createSlice(ctx context.Context, array llvm.LLVMValueRef, elementType llvm.LLVMTypeRef, numElements uint64, low, high, max llvm.LLVMValueRef) llvm.LLVMValueRef {
	stringType := c.createRuntimeType(ctx, "_string").valueType
	sliceType := c.createRuntimeType(ctx, "_slice").valueType

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

	// Cast integer types
	lengthVal = c.castInt(ctx, lengthVal, llvm.Int32TypeInContext(c.currentContext(ctx)))
	capacityVal = c.castInt(ctx, capacityVal, llvm.Int32TypeInContext(c.currentContext(ctx)))
	low = c.castInt(ctx, low, llvm.Int32TypeInContext(c.currentContext(ctx)))
	high = c.castInt(ctx, low, llvm.Int32TypeInContext(c.currentContext(ctx)))
	max = c.castInt(ctx, low, llvm.Int32TypeInContext(c.currentContext(ctx)))

	// Create the runtime call
	value := c.createRuntimeCall(ctx, "sliceAddr", []llvm.LLVMValueRef{
		ptrVal, lengthVal, capacityVal, elementSizeVal, low, high, max,
	})

	// Return a new string if the input was a string
	if isString {
		ptrVal = llvm.BuildExtractValue(c.builder, value, 0, "")
		lengthVal = llvm.BuildExtractValue(c.builder, value, 1, "")

		// Create a new string
		value = c.createAlloca(ctx, stringType, "")
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

	var array llvm.LLVMValueRef
	if len(values) > 0 {
		array = c.createArrayAlloca(ctx, lastType, uint64(len(values)), "")

		// Populate the underlying array for the slice
		for i, value := range values {
			index := llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(i), false)
			addr := llvm.BuildGEP2(c.builder, lastType, array, []llvm.LLVMValueRef{index}, "")
			llvm.BuildStore(c.builder, value, addr)
		}
	} else {
		array = c.ptrType.Nil()
	}

	// Create the slice struct
	sliceType := c.createRuntimeType(ctx, "_slice").valueType
	result := c.createAlloca(ctx, sliceType, "")

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
	sliceType := c.createRuntimeType(ctx, "_slice").valueType
	stringType := c.createRuntimeType(ctx, "_string").valueType

	if !llvm.TypeIsEqual(stringType, llvm.TypeOf(str)) {
		panic("value is not a string")
	}

	// Create the slice struct
	slice := c.createAlloca(ctx, sliceType, "")

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

func (c *Compiler) makeInterface(ctx context.Context, value ssa.Value) llvm.LLVMValueRef {
	x := c.createExpression(ctx, value).UnderlyingValue(ctx)

	interfaceType := c.createRuntimeType(ctx, "_interface").valueType

	// Return the value if it is already of interface type
	if llvm.TypeIsEqual(llvm.TypeOf(x), interfaceType) {
		return x
	}

	if llvm.GetTypeKind(llvm.TypeOf(x)) != llvm.PointerTypeKind {
		x = c.addressOf(ctx, x)
	}

	typeinfo := c.createTypeDescriptor(ctx, c.createType(ctx, value.Type().Underlying()))
	args := []llvm.LLVMValueRef{
		x,
		typeinfo,
	}

	return c.createRuntimeCall(ctx, "interfaceMake", args)
}

func (c *Compiler) addressOf(ctx context.Context, value llvm.LLVMValueRef) llvm.LLVMValueRef {
	alloca := c.createAlloca(ctx, llvm.TypeOf(value), "")
	llvm.BuildStore(c.builder, value, alloca)
	return alloca
}

func (c *Compiler) createConstantString(ctx context.Context, str string) llvm.LLVMValueRef {
	// Get the string type
	strType := c.createRuntimeType(ctx, "_string").valueType
	if strType == nil {
		panic("missing string type")
	}

	var strArrVal llvm.LLVMValueRef
	if len(str) > 0 {
		//cstr := llvm.ConstStringInContext(c.currentContext(ctx), str, true)
		//strArrVal = c.createGlobalValue(ctx, cstr, c.symbolName(c.currentPackage(ctx).Pkg, "cstring"))
		// Create the global value, but don't initialize it yet. The constant string value will be placed into string
		// table in order de-duplicate strings in order to save program memory.
		strArrVal = llvm.AddGlobal(
			c.module,
			c.ptrType.valueType,
			c.symbolName(c.currentPackage(ctx).Pkg, "cstring_ptr"))

		// Map the global to the string value
		c.stringTable[str] = append(c.stringTable[str], llvm.GetValueName2(strArrVal))
	} else {
		strArrVal = llvm.ConstNull(llvm.StructGetTypeAtIndex(strType, 0))
	}

	// Return a string struct
	return llvm.ConstNamedStruct(
		strType,
		[]llvm.LLVMValueRef{
			strArrVal,
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(str)), false),
		})
}

func (c *Compiler) createGlobalString(ctx context.Context, str string) llvm.LLVMValueRef {
	strVal := c.createConstantString(ctx, str)
	return c.createGlobalValue(ctx, strVal, c.symbolName(c.currentPackage(ctx).Pkg, "gostring"))
}

func (c *Compiler) createGlobalValue(ctx context.Context, constVal llvm.LLVMValueRef, name string) llvm.LLVMValueRef {
	if !llvm.IsConstant(constVal) {
		panic("attempted to create global from non-const value")
	}

	// Create the global that will hold the constant string value's address
	value := llvm.AddGlobal(c.module, llvm.TypeOf(constVal), name)

	// Set the global variable's value
	llvm.SetInitializer(value, constVal)
	llvm.SetUnnamedAddr(value, true)
	//value = llvm.BuildBitCast(c.builder, value, c.ptrType.valueType, "")
	return value
}

func (c *Compiler) structFieldAddress(ctx context.Context, value Value, structType llvm.LLVMTypeRef, index int) (llvm.LLVMTypeRef, llvm.LLVMValueRef) {
	// Get the type of the field
	fieldType := llvm.StructGetTypeAtIndex(structType, uint(index))

	// Create a GEP to get the  address of the field in the struct
	return fieldType, llvm.BuildStructGEP2(c.builder, structType, value.UnderlyingValue(ctx), uint(index), "")
}

func (c *Compiler) createAlloca(ctx context.Context, _type llvm.LLVMTypeRef, name string) (result llvm.LLVMValueRef) {
	// Create the alloca in the current function's entry block
	c.positionAtEntryBlock(ctx)
	result = llvm.BuildAlloca(c.builder, _type, strings.TrimRight(strings.Join([]string{"stack_var", name}, "_"), "_"))
	llvm.PositionBuilderAtEnd(c.builder, c.currentBlock(ctx))
	return
}

func (c *Compiler) createArrayAlloca(ctx context.Context, _type llvm.LLVMTypeRef, length uint64, name string) (result llvm.LLVMValueRef) {
	// Create the alloca in the current function's entry block
	c.positionAtEntryBlock(ctx)
	lengthVal := llvm.ConstInt(c.int32Type(ctx), length, false)
	result = llvm.BuildArrayAlloca(c.builder, _type, lengthVal, strings.TrimRight(strings.Join([]string{"stack_arr_var", name}, "_"), "_"))
	llvm.PositionBuilderAtEnd(c.builder, c.currentBlock(ctx))
	return
}

func (c *Compiler) castInt(ctx context.Context, value llvm.LLVMValueRef, to llvm.LLVMTypeRef) llvm.LLVMValueRef {
	valueType := llvm.TypeOf(value)
	if llvm.GetTypeKind(valueType) != llvm.IntegerTypeKind {
		panic("value is not an integer type")
	}

	if llvm.GetTypeKind(to) != llvm.IntegerTypeKind {
		panic("cannot cast to non-integer type")
	}

	if valueType == to {
		return value
	}

	isSigned := false
	switch valueType {
	case llvm.Int8TypeInContext(c.currentContext(ctx)),
		llvm.Int16TypeInContext(c.currentContext(ctx)),
		llvm.Int32TypeInContext(c.currentContext(ctx)),
		llvm.Int64TypeInContext(c.currentContext(ctx)):
		isSigned = true
	}

	return llvm.BuildIntCast2(c.builder, value, to, isSigned, "")
}

func (c *Compiler) createConstBool(ctx context.Context, value bool) llvm.LLVMValueRef {
	if value {
		return llvm.ConstInt(c.int1Type(ctx), 1, false)
	}
	return llvm.ConstInt(c.int1Type(ctx), 0, false)
}
