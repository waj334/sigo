package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

type ConstructType int

const (
	InvalidConstructType ConstructType = iota
	Primitive
	Pointer
	Interface
	Struct
	Array
	Slice
	Map
	Channel
)

func (c *Compiler) createTypeDescriptor(ctx context.Context, typ *Type) (descriptor llvm.LLVMValueRef) {
	// Return the previously create type descriptor
	if descriptor, ok := c.descriptors[typ.spec]; ok {
		return descriptor
	}

	c.printf(Debug, "Creating type descriptor for %s\n", typ.spec.String())
	defer c.printf(Debug, "Done creating type descriptor for %s\n", typ.spec.String())

	goType := typ.spec

	// Find the "typeDescriptor" runtime type
	descriptorType := c.createRuntimeType(ctx, "_type").valueType
	if descriptorType != nil {
		// Create an array to store the typeDescriptor struct values
		var descriptorValues [11]llvm.LLVMValueRef
		var name string
		var construct ConstructType

		// Extract information from the type before processing
		switch goType := goType.(type) {
		case *types.Named:
			name = goType.Obj().Id()
		case *types.Pointer:
			name = goType.String()
		}

		// Store the size of this type in bytes
		descriptorValues[1] = llvm.ConstInt(
			c.uintptrType.valueType,
			llvm.StoreSizeOfType(c.options.Target.dataLayout, typ.valueType), false)

		switch goType := goType.Underlying().(type) {
		case *types.Basic:
			construct = Primitive
			// Store the name of the basic type if no name has been determined
			if len(name) == 0 {
				name = goType.Name()
			}
			// Store the Go type kind
			descriptorValues[3] = llvm.ConstInt(
				llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(goType.Kind()), false)
		case *types.Chan:
			construct = Channel
			chanType := c.createRuntimeType(ctx, "_channelType").valueType
			descriptorValues[9] = c.createGlobalValue(ctx,
				llvm.ConstNamedStruct(chanType, []llvm.LLVMValueRef{
					c.createTypeDescriptor(ctx, c.createType(ctx, goType.Elem())),
					llvm.ConstInt(c.int32Type(ctx), uint64(goType.Dir()), false),
				}), "chan_type")
		case *types.Interface:
			construct = Interface
			// Collect the interface's methods
			var methods []*types.Func
			for i := 0; i < goType.NumMethods(); i++ {
				methods = append(methods, goType.Method(i))
			}
			// Create the method table struct
			descriptorValues[4] = c.createMethodTable(ctx, methods)
		case *types.Struct:
			construct = Struct
			// Find all methods for this type in the package
			if named, ok := typ.spec.(*types.Named); ok {
				// Collect this type's method
				var methods []*types.Func
				for i := 0; i < named.NumMethods(); i++ {
					methods = append(methods, named.Method(i))
				}
				// Create the method table struct
				descriptorValues[4] = c.createMethodTable(ctx, methods)
			}
		case *types.Pointer:
			construct = Pointer
			// Create the descriptor for the element type
			elementDescriptor := c.createTypeDescriptor(ctx, c.createType(ctx, goType.Elem()))

			// Create the pointer value descriptor
			pointerDescType := c.createRuntimeType(ctx, "_pointerType").valueType
			descriptorValues[8] = c.createGlobalValue(ctx,
				llvm.ConstNamedStruct(pointerDescType, []llvm.LLVMValueRef{elementDescriptor}),
				c.symbolName(c.currentPackage(ctx).Pkg, "pointer_type"))
		case *types.Array:
			descriptorValues[6] = c.createArrayDescriptor(ctx, goType.Elem(), goType.Len())
		case *types.Slice:
			descriptorValues[6] = c.createArrayDescriptor(ctx, goType.Elem(), -1)
		}

		// Create a global for the typename
		descriptorValues[0] = c.createGlobalString(ctx, name)

		// Store the construct type
		descriptorValues[2] = llvm.ConstInt(
			llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(construct), false)

		// Fill nil pointers
		members := llvm.GetStructElementTypes(descriptorType)

		for i := 0; i < len(members); i++ {
			if descriptorValues[i] == nil {
				descriptorValues[i] = llvm.ConstNull(members[i])
			}
			if !llvm.TypeIsEqual(llvm.TypeOf(descriptorValues[i]), members[i]) {
				panic("types do not match")
			}
		}

		// Create a const descriptor struct value
		descriptor = c.createGlobalValue(ctx,
			llvm.ConstNamedStruct(descriptorType, descriptorValues[:]),
			c.symbolName(c.currentPackage(ctx).Pkg, "type"))
		c.descriptors[typ.spec] = descriptor
	}
	return
}

func (c *Compiler) createMethodTable(ctx context.Context, methods []*types.Func) llvm.LLVMValueRef {
	// Get the required types
	methodTableType := c.createRuntimeType(ctx, "_methodTable").valueType
	if methodTableType == nil {
		panic("missing runtime._methodTable type")
	}

	methodTableValues := []llvm.LLVMValueRef{
		llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(methods)), false),
		llvm.ConstNull(c.ptrType.valueType),
	}

	if len(methods) > 0 {
		// Create a const array representing the method map entries
		methodValues := make([]llvm.LLVMValueRef, 0, len(methods))
		for _, method := range methods {
			// Create the function descriptor struct
			methodValues = append(methodValues, c.createFunctionDescriptor(ctx, method))
		}

		// Create the array of method descriptors
		methodTableValues[1] = c.createGlobalValue(ctx,
			llvm.ConstArray(c.ptrType.valueType, methodValues),
			c.symbolName(c.currentPackage(ctx).Pkg, "methods"))
	}

	// Create the method table struct
	descriptor := c.createGlobalValue(ctx,
		llvm.ConstNamedStruct(methodTableType, methodTableValues),
		c.symbolName(c.currentPackage(ctx).Pkg, "methodTable"))

	return descriptor
}

func (c *Compiler) createFunctionDescriptor(ctx context.Context, fn *types.Func) llvm.LLVMValueRef {
	if descriptor, ok := c.descriptors[fn.Type()]; ok {
		return descriptor
	} else {
		descriptorType := c.createRuntimeType(ctx, "_type").valueType
		if descriptorType == nil {
			panic("missing Type type")
		}

		funcDescriptorType := c.createRuntimeType(ctx, "_funcType").valueType
		if funcDescriptorType == nil {
			panic("missing FunctionDescriptor type")
		}

		tableType := c.createRuntimeType(ctx, "_typeTable").valueType
		if tableType == nil {
			panic("missing DescriptorTable type")
		}

		signature := fn.Type().(*types.Signature)

		// Collect argument types
		argTypes := make([]llvm.LLVMValueRef, 0, signature.Params().Len())
		for i := 0; i < signature.Params().Len(); i++ {
			// Create the type descriptor value
			argTypeDescriptor := c.createTypeDescriptor(ctx, c.createType(ctx, signature.Params().At(i).Type()))

			// Store a pointer to it in the "argTypes" slice defined above
			argTypes = append(argTypes, argTypeDescriptor)
		}

		// Create the arg table
		argTableValues := [2]llvm.LLVMValueRef{
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(argTypes)), false),
		}

		if len(argTypes) > 0 {
			argTableValues[1] = c.createGlobalValue(ctx,
				llvm.ConstArray(c.ptrType.valueType, argTypes),
				c.symbolName(c.currentPackage(ctx).Pkg, "args"))
		} else {
			argTableValues[1] = llvm.ConstNull(c.ptrType.valueType)
		}

		argTable := llvm.ConstNamedStruct(tableType, argTableValues[:])

		// Collect return types
		returnTypes := make([]llvm.LLVMValueRef, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			// Create the type descriptor value
			returnTypeDescriptor := c.createTypeDescriptor(ctx, c.createType(ctx, signature.Results().At(i).Type()))

			// Store a pointer to it in the "argTypes" slice defined above
			returnTypes = append(returnTypes, returnTypeDescriptor)
		}

		// Create the return table
		arr := c.createGlobalValue(ctx,
			llvm.ConstArray(c.ptrType.valueType, returnTypes),
			c.symbolName(c.currentPackage(ctx).Pkg, "returns"))

		returnTable := llvm.ConstNamedStruct(tableType, []llvm.LLVMValueRef{
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(returnTypes)), false),
			arr,
		})

		var fnValue llvm.LLVMValueRef
		ssaFn := c.currentPackage(ctx).Prog.FuncValue(fn)
		if ssaFn != nil {
			fnValue = c.createExpression(ctx, ssaFn).UnderlyingValue(ctx)
		} else {
			fnValue = llvm.ConstNull(c.ptrType.valueType)
		}

		// Collect the values for the functionDescriptor struct
		funcDescriptorValues := []llvm.LLVMValueRef{
			llvm.BuildBitCast(c.builder, fnValue, c.ptrType.valueType, ""),
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(c.computeFunctionHash(fn)), false),
			c.createGlobalString(ctx, fn.Name()),
			c.createGlobalValue(ctx, argTable, c.symbolName(c.currentPackage(ctx).Pkg, "args")),
			c.createGlobalValue(ctx, returnTable, c.symbolName(c.currentPackage(ctx).Pkg, "returns")),
		}

		value := c.createGlobalValue(ctx,
			llvm.ConstNamedStruct(funcDescriptorType, funcDescriptorValues),
			c.symbolName(c.currentPackage(ctx).Pkg, "funcType"))

		// Cache this descriptor for fast lookup later
		c.descriptors[fn.Type()] = value
		return value
	}
}

func (c *Compiler) createArrayDescriptor(ctx context.Context, elem types.Type, length int64) llvm.LLVMValueRef {
	desc := c.createRuntimeType(ctx, "_arrayType").valueType
	if desc == nil {
		panic("missing _arrayType type")
	}

	// Create the descriptor for the element type
	elemType := c.createTypeDescriptor(ctx, c.createType(ctx, elem))

	// Create the struct
	arr := llvm.ConstNamedStruct(desc, []llvm.LLVMValueRef{
		elemType,
		llvm.ConstInt(c.int32Type(ctx), uint64(length), false),
	})

	return c.createGlobalValue(ctx, arr, c.symbolName(c.currentPackage(ctx).Pkg, "arr_type"))
}
