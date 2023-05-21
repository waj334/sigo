package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createType(ctx context.Context, typ types.Type) *Type {
	// Check if this type was already created
	if result, ok := c.types[typ]; ok {
		c.printf(Debug, "Returning type from cache: %s\n", typ.String())
		return result
	}

	typename := typ.String()
	c.printf(Debug, "Creating type: %s\n", typename)

	result := &Type{}

	// Create the equivalent LLVM type
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
		case types.Int8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
		case types.Uint16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
		case types.Int16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
		case types.Int, types.Int32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
		case types.Uint, types.Uint32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
		case types.Uint64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
		case types.Int64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
		case types.Uintptr:
			result.valueType = c.uintptrType.valueType
		case types.UnsafePointer:
			result.valueType = c.ptrType.valueType
		case types.Float32:
			result.valueType = llvm.FloatTypeInContext(c.currentContext(ctx))
		case types.Float64:
			result.valueType = llvm.DoubleTypeInContext(c.currentContext(ctx))
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String:
			result.valueType = llvm.GetTypeByName2(c.currentContext(ctx), "string")
		case types.Bool:
			result.valueType = llvm.Int1TypeInContext(c.currentContext(ctx))
		default:
			panic("encountered unknown basic type")
		}
	case *types.Struct:
		var structMembers []llvm.LLVMTypeRef
		for i := 0; i < typ.NumFields(); i++ {
			fieldType := c.createType(ctx, typ.Field(i).Type())
			structMembers = append(structMembers, fieldType.valueType)
		}
		result.valueType = llvm.StructTypeInContext(c.currentContext(ctx), structMembers, false)
	case *types.Map:
		result.valueType = llvm.GetTypeByName2(c.currentContext(ctx), "map")
	case *types.Slice:
		result.valueType = llvm.GetTypeByName2(c.currentContext(ctx), "slice")
	case *types.Named:
		if structType, ok := typ.Underlying().(*types.Struct); ok {
			// Create a named struct with the same body as the underlying struct type
			result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), typ.Obj().Id())

			// Note: Need to cache structs right away to prevent a stack overflow
			// with a member type is or contains this struct type for any reason.
			c.types[typ] = result

			// Create the underlying struct type
			st := c.createType(ctx, structType)

			// Set the struct body to that of the underlying struct type
			llvm.StructSetBody(result.valueType, llvm.GetStructElementTypes(st.valueType), false)
		}
		result = c.createType(ctx, typ.Underlying())
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.ArrayType(elementType.valueType, uint(typ.Len()))
	case *types.Interface:
		result.valueType = llvm.GetTypeByName2(c.currentContext(ctx), "interface")
	case *types.Pointer:
		if structType, ok := typ.Elem().Underlying().(*types.Struct); ok {
			name := ""
			if named, ok := typ.Elem().(*types.Named); ok {
				name = named.Obj().Name()
			}

			// Create a named struct with the same body as the underlying struct type
			result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)

			// Note: Need to cache structs right away to prevent a stack overflow
			// with a member type is or contains this struct type for any reason.
			c.types[typ] = result

			// Create the underlying struct type
			st := c.createType(ctx, structType)

			// Set the struct body to that of the underlying struct type
			llvm.StructSetBody(result.valueType, llvm.GetStructElementTypes(st.valueType), false)
		}
		elementType := c.createType(ctx, typ.Elem())
		// NOTE: This pointer is opaque!!!
		result.valueType = llvm.PointerType(elementType.valueType, 0)
	case *types.Chan:
		result.valueType = llvm.GetTypeByName2(c.currentContext(ctx), "channel")
	case *types.Signature:
		var returnValueTypes []llvm.LLVMTypeRef
		var argValueTypes []llvm.LLVMTypeRef
		var returnType llvm.LLVMTypeRef

		if numArgs := typ.Results().Len(); numArgs == 0 {
			returnType = llvm.VoidTypeInContext(c.currentContext(ctx))
		} else if numArgs == 1 {
			returnType = c.createType(ctx, typ.Results().At(0).Type()).valueType
		} else {
			// Create a struct type to store the return values into
			for i := 0; i < numArgs; i++ {
				resultType := typ.Results().At(i).Type()
				returnValueTypes = append(returnValueTypes, c.createType(ctx, resultType).valueType)
			}
			returnType = llvm.StructTypeInContext(c.currentContext(ctx), returnValueTypes, false)
		}

		// Create types for the arguments
		for i := 0; i < typ.Params().Len(); i++ {
			arg := typ.Params().At(i)
			argType := c.createType(ctx, arg.Type())
			argValueTypes = append(argValueTypes, argType.valueType)
		}

		// Create the function type
		fnType := llvm.FunctionType(returnType, argValueTypes, typ.Variadic())
		result.valueType = llvm.PointerType(fnType, 0)

		// Track the element type of this pointer
		c.signatures[typ] = fnType
	case *types.Tuple:
		var memberTypes []llvm.LLVMTypeRef
		if typ != nil {
			for i := 0; i < typ.Len(); i++ {
				memberTypes = append(memberTypes, c.createType(ctx, typ.At(i).Type()).valueType)
			}
		}
		result.valueType = llvm.StructTypeInContext(c.currentContext(ctx), memberTypes, false)
	default:
		panic("encountered unknown type")
	}

	// A known type should be created under all circumstances
	if result.valueType == nil {
		panic("no type was created")
	}

	// Store the go type
	result.spec = typ

	// Cache the type for faster lookup later
	c.types[typ] = result

	// Return the type
	return result
}
