package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createTypeDescriptor(ctx context.Context, typ Type) (descriptor llvm.LLVMValueRef) {
	// Return the previously create type descriptor
	if descriptor, ok := c.descriptors[typ.spec]; ok {
		return descriptor
	}

	goType := typ.spec

	// Find the "typeDescriptor" runtime type
	descriptorType, ok := c.findRuntimeType(ctx, "runtime/internal/go.typeDescriptor")
	if ok {
		// Create an array to store the typeDescriptor struct's values
		var descriptorValues [10]llvm.LLVMValueRef

		name := ""
		// Get the type's name if it is a named type
		if namedType, ok := goType.(*types.Named); ok {
			name = namedType.Obj().Id()

			// Begin processing the underlying type below
			goType = namedType.Underlying()
		}

		// Create a global for the typename
		descriptorValues[0] = c.createGlobalString(ctx, name)

		// Store the size of this type in bytes
		descriptorValues[1] = llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), llvm.StoreSizeOfType(c.options.Target.dataLayout, typ.valueType), false)

		switch goType := goType.(type) {
		case *types.Basic:
			// Store the Go type kind
			descriptorValues[2] = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(goType.Kind()), false)
		case *types.Interface:
			// Collect the interface's methods
			var methods []*types.Func
			for i := 0; i < goType.NumMethods(); i++ {
				methods = append(methods, goType.Method(i))
			}

			// Create the method table struct
			descriptorValues[3] = c.createMethodTable(ctx, methods)
		}

		// Fill nil pointers
		members := llvm.GetStructElementTypes(descriptorType.valueType)

		for i := 0; i < len(members); i++ {
			if descriptorValues[i] == nil {
				descriptorValues[i] = llvm.ConstNull(members[i])
			}
			if !llvm.TypeIsEqual(llvm.TypeOf(descriptorValues[i]), members[i]) {
				panic("types do not match")
			}
		}

		// Create a const descriptor struct value
		strDescriptorVal := llvm.ConstNamedStruct(descriptorType.valueType, descriptorValues[:])

		// Create a const typeDescriptor global for the input type
		descriptor = llvm.AddGlobal(c.module, descriptorType.valueType, "typeDescriptor")

		// Set the global variable's value
		llvm.SetInitializer(descriptor, strDescriptorVal)
		llvm.SetUnnamedAddr(descriptor, llvm.GlobalUnnamedAddr != 0)

		descriptor = llvm.BuildBitCast(c.builder, descriptor, c.ptrType.valueType, "")
		c.descriptors[typ.spec] = descriptor
	}
	return
}

func (c *Compiler) createMethodTable(ctx context.Context, methods []*types.Func) llvm.LLVMValueRef {
	// Get the required types
	methodTableType, ok := c.findRuntimeType(ctx, "runtime/internal/go.methodTable")
	if !ok {
		panic("runtime missing methodTable type")
	}

	var methodTableValues [2]llvm.LLVMValueRef

	// Set the method count
	methodTableValues[0] = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(methods)), false)

	// Create a const array representing the method map entries
	methodValues := make([]llvm.LLVMValueRef, 0, len(methods))
	for _, method := range methods {
		// Create the function descriptor struct
		methodValues = append(methodValues, c.createFunctionDescriptor(ctx, method))
	}

	// Create the array of method descriptors
	methodTableValues[1] = c.createGlobalValue(ctx, llvm.ConstArray2(c.ptrType.valueType, methodValues), "methods")

	// Create the method table struct
	value := llvm.ConstNamedStruct(methodTableType.valueType, methodTableValues[:])

	// Create a const typeDescriptor global for the input type
	descriptor := llvm.AddGlobal(c.module, methodTableType.valueType, "methodTable")

	// Set the global variable's value
	llvm.SetInitializer(descriptor, value)
	llvm.SetUnnamedAddr(descriptor, llvm.GlobalUnnamedAddr != 0)

	descriptor = llvm.BuildBitCast(c.builder, descriptor, c.ptrType.valueType, "")
	return descriptor
}

func (c *Compiler) createFunctionDescriptor(ctx context.Context, fn *types.Func) llvm.LLVMValueRef {
	if descriptor, ok := c.descriptors[fn.Type()]; ok {
		return descriptor
	} else {
		funcDescriptorType, ok := c.findRuntimeType(ctx, "runtime/internal/go.functionDescriptor")
		if !ok {
			panic("runtime missing functionDescriptor type")
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

		// Collect return types
		returnTypes := make([]llvm.LLVMValueRef, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			// Create the type descriptor value
			returnTypeDescriptor := c.createTypeDescriptor(ctx, c.createType(ctx, signature.Results().At(i).Type()))

			// Store a pointer to it in the "argTypes" slice defined above
			returnTypes = append(returnTypes, returnTypeDescriptor)
		}

		// Collect the values for the functionDescriptor struct
		funcDescriptorValues := []llvm.LLVMValueRef{
			c.createGlobalString(ctx, fn.Name()),
			llvm.ConstInt(llvm.Int32Type(), uint64(signature.Params().Len()), false),
			c.createGlobalValue(ctx, llvm.ConstArray2(c.ptrType.valueType, argTypes), "params"),
			llvm.ConstInt(llvm.Int32Type(), uint64(signature.Results().Len()), false),
			c.createGlobalValue(ctx, llvm.ConstArray2(c.ptrType.valueType, returnTypes), "returns"),
		}

		value := llvm.ConstNamedStruct(funcDescriptorType.valueType, funcDescriptorValues)

		// Create a const typeDescriptor global for the input type
		descriptor = llvm.AddGlobal(c.module, funcDescriptorType.valueType, "functionDescriptor")

		// Set the global variable's value
		llvm.SetInitializer(descriptor, value)
		llvm.SetUnnamedAddr(descriptor, llvm.GlobalUnnamedAddr != 0)

		// Cast the descriptor to the pointer type for this target
		descriptor = llvm.BuildBitCast(c.builder, descriptor, c.ptrType.valueType, "")

		// Cache this descriptor for fast lookup later
		c.descriptors[fn.Type()] = descriptor
		return descriptor
	}
}

func (c *Compiler) createGlobalString(ctx context.Context, str string) llvm.LLVMValueRef {
	// Create a constant string value
	constVal := llvm.BuildGlobalStringPtr(c.builder, str, "")
	constVal = llvm.BuildBitCast(c.builder, constVal, c.ptrType.valueType, "")

	// Create a string struct
	strType, ok := c.findRuntimeType(ctx, "runtime/internal/go.stringDescriptor")
	if !ok {
		panic("runtime missing stringDescriptor type")
	}

	strVal := llvm.ConstNamedStruct(strType.valueType, []llvm.LLVMValueRef{
		constVal,
		llvm.ConstInt(llvm.Int32Type(), uint64(len(str)), false),
	})

	// Create the global that will hold the constant string value's address
	value := llvm.AddGlobal(c.module, strType.valueType, "global_string")

	// Set the global variable's value
	llvm.SetInitializer(value, strVal)
	llvm.SetUnnamedAddr(value, llvm.GlobalUnnamedAddr != 0)

	// Bit cast the value
	value = llvm.BuildBitCast(c.builder, value, c.ptrType.valueType, "")

	return value
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
