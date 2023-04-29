package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createTypeInfoTypes(ctx llvm.LLVMContextRef) {
	typeType := llvm.StructCreateNamed(ctx, "Type")
	typeTable := llvm.StructCreateNamed(ctx, "DescriptorTable")
	typeFunctionDesc := llvm.StructCreateNamed(ctx, "FunctionDescriptor")
	typeFieldDesc := llvm.StructCreateNamed(ctx, "FieldDescriptor")
	typeArrayDesc := llvm.StructCreateNamed(ctx, "ArrayDescriptor")
	typeMapDesc := llvm.StructCreateNamed(ctx, "MapDescriptor")
	typePointerDesc := llvm.StructCreateNamed(ctx, "PointerDescriptor")
	typeChannelDesc := llvm.StructCreateNamed(ctx, "ChannelDescriptor")

	// Type struct body
	llvm.StructSetBody(typeType, []llvm.LLVMTypeRef{
		llvm.PointerType(llvm.GetTypeByName2(ctx, "string"), 0),
		llvm.Int64TypeInContext(ctx),
		llvm.Int32TypeInContext(ctx),
		llvm.PointerType(typeTable, 0),
		llvm.PointerType(typeTable, 0),
		llvm.PointerType(typeArrayDesc, 0),
		llvm.PointerType(typeMapDesc, 0),
		llvm.PointerType(typePointerDesc, 0),
		llvm.PointerType(typeChannelDesc, 0),
		llvm.PointerType(typeFunctionDesc, 0),
	}, false)

	// Table struct body
	llvm.StructSetBody(typeTable, []llvm.LLVMTypeRef{
		llvm.Int32TypeInContext(ctx),
		c.ptrType.valueType,
	}, false)

	// FunctionDescriptor struct body
	llvm.StructSetBody(typeFunctionDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(llvm.GetTypeByName2(ctx, "string"), 0),
		typeTable,
		typeTable,
	}, false)

	// FieldDescriptor struct body
	llvm.StructSetBody(typeFieldDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(llvm.GetTypeByName2(ctx, "string"), 0),
		typeType,
		llvm.PointerType(llvm.GetTypeByName2(ctx, "string"), 0),
	}, false)

	// ArrayDescriptor struct body
	llvm.StructSetBody(typeArrayDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(typeType, 0),
		llvm.Int32TypeInContext(ctx),
		llvm.Int32TypeInContext(ctx),
	}, false)

	// MapDescriptor struct body
	llvm.StructSetBody(typeMapDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(typeType, 0),
		llvm.PointerType(typeType, 0),
	}, false)

	// PointerDescriptor struct body
	llvm.StructSetBody(typePointerDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(typeType, 0),
	}, false)

	// ChannelDescriptor struct body
	llvm.StructSetBody(typeChannelDesc, []llvm.LLVMTypeRef{
		llvm.PointerType(typeType, 0),
		llvm.Int32TypeInContext(ctx),
		llvm.Int32TypeInContext(ctx),
	}, false)

	return
}

func (c *Compiler) createTypeDescriptor(ctx context.Context, typ *Type) (descriptor llvm.LLVMValueRef) {
	// Return the previously create type descriptor
	if descriptor, ok := c.descriptors[typ.spec]; ok {
		return descriptor
	}

	goType := typ.spec

	// Find the "typeDescriptor" runtime type
	descriptorType := llvm.GetTypeByName2(c.currentContext(ctx), "Type")
	if descriptorType != nil {
		// Create an array to store the typeDescriptor struct's values
		var descriptorValues [10]llvm.LLVMValueRef

		var name string
		// Get the type's name if it is a named type
		if namedType, ok := goType.(*types.Named); ok {
			name = namedType.Obj().Id()

			// Begin processing the underlying type below
			goType = namedType.Underlying()
		}

		// Store the size of this type in bytes
		descriptorValues[1] = llvm.ConstInt(
			llvm.Int64TypeInContext(c.currentContext(ctx)),
			llvm.StoreSizeOfType(c.options.Target.dataLayout, typ.valueType), false)

		switch goType := goType.Underlying().(type) {
		case *types.Basic:
			// Store the name of the basic type if no name has been determined
			if len(name) == 0 {
				name = goType.Name()
			}

			// Store the Go type kind
			descriptorValues[2] = llvm.ConstInt(
				llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(goType.Kind()), false)
		case *types.Interface:
			// Collect the interface's methods
			var methods []*types.Func
			for i := 0; i < goType.NumMethods(); i++ {
				methods = append(methods, goType.Method(i))
			}

			// Create the method table struct
			descriptorValues[3] = c.createMethodTable(ctx, methods)
		}

		// Create a global for the typename
		descriptorValues[0] = c.createGlobalString(ctx, name)

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
			llvm.ConstNamedStruct(descriptorType, descriptorValues[:]), "typeDescriptor")
		c.descriptors[typ.spec] = descriptor
	}
	return
}

func (c *Compiler) createMethodTable(ctx context.Context, methods []*types.Func) llvm.LLVMValueRef {
	// Get the required types
	methodTableType := llvm.GetTypeByName2(c.currentContext(ctx), "DescriptorTable")
	if methodTableType == nil {
		panic("missing DescriptorTable type")
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
			llvm.ConstArray(c.ptrType.valueType, methodValues), "methods")
	}

	// Create the method table struct
	descriptor := c.createGlobalValue(ctx,
		llvm.ConstNamedStruct(methodTableType, methodTableValues), "methodTable")

	return descriptor
}

func (c *Compiler) createFunctionDescriptor(ctx context.Context, fn *types.Func) llvm.LLVMValueRef {
	if descriptor, ok := c.descriptors[fn.Type()]; ok {
		return descriptor
	} else {
		descriptorType := llvm.GetTypeByName2(c.currentContext(ctx), "Type")
		if descriptorType == nil {
			panic("missing Type type")
		}

		funcDescriptorType := llvm.GetTypeByName2(c.currentContext(ctx), "FunctionDescriptor")
		if funcDescriptorType == nil {
			panic("missing FunctionDescriptor type")
		}

		tableType := llvm.GetTypeByName2(c.currentContext(ctx), "DescriptorTable")
		if funcDescriptorType == nil {
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
		argTable := llvm.ConstNamedStruct(tableType, []llvm.LLVMValueRef{
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(argTypes)), false),
			llvm.ConstArray(descriptorType, argTypes),
		})

		// Collect return types
		returnTypes := make([]llvm.LLVMValueRef, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			// Create the type descriptor value
			returnTypeDescriptor := c.createTypeDescriptor(ctx, c.createType(ctx, signature.Results().At(i).Type()))

			// Store a pointer to it in the "argTypes" slice defined above
			returnTypes = append(returnTypes, returnTypeDescriptor)
		}

		// Create the return table
		returnTable := llvm.ConstNamedStruct(tableType, []llvm.LLVMValueRef{
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(returnTypes)), false),
			llvm.ConstArray(descriptorType, returnTypes),
		})

		// Collect the values for the functionDescriptor struct
		funcDescriptorValues := []llvm.LLVMValueRef{
			c.createGlobalString(ctx, fn.Name()),
			c.createGlobalValue(ctx, argTable, "args"),
			c.createGlobalValue(ctx, returnTable, "returns"),
		}

		value := c.createGlobalValue(ctx,
			llvm.ConstNamedStruct(funcDescriptorType, funcDescriptorValues), "functionDescriptor")

		// Cache this descriptor for fast lookup later
		c.descriptors[fn.Type()] = value
		return value
	}
}
