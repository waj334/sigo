package compiler

import (
	"context"
	"go/types"
	"strings"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createType(ctx context.Context, typ types.Type) *Type {
	typename := typ.String()

	// Check if this type was already created
	if result, ok := c.types[typ]; ok {
		c.printf(Debug, "Returning type from cache: %s\n", typename)
		return result
	}

	c.printf(Debug, "Creating type: %s\n", typename)

	result := &Type{
		spec: typ,
	}

	// Create the equivalent LLVM type
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8, types.Int8:
			result.valueType = c.int8Type(ctx)
		case types.Uint16, types.Int16:
			result.valueType = c.int16Type(ctx)
		case types.Int, types.Int32, types.Uint, types.Uint32:
			result.valueType = c.int32Type(ctx)
		case types.Uint64, types.Int64:
			result.valueType = c.int64Type(ctx)
		case types.Uintptr:
			result.valueType = c.uintptrType.valueType
		case types.UnsafePointer:
			result.valueType = c.ptrType.valueType
		case types.Float32:
			result.valueType = c.float32(ctx)
		case types.Float64:
			result.valueType = c.float64(ctx)
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String, types.UntypedString:
			tt := c.createRuntimeType(ctx, "_string")
			result.valueType = tt.valueType
			result.debugType = c.createDebugType(ctx, tt.spec)
			tt.spec = typ
		case types.Bool, types.UntypedBool:
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
		tt := c.createRuntimeType(ctx, "_map")
		result.valueType = tt.valueType
		result.debugType = c.createDebugType(ctx, tt.spec)
		tt.spec = typ
	case *types.Slice:
		tt := c.createRuntimeType(ctx, "_slice")
		result.valueType = tt.valueType
		result.debugType = c.createDebugType(ctx, tt.spec)
		tt.spec = typ
	case *types.Named:
		if structType, ok := typ.Underlying().(*types.Struct); ok {
			// Handle runtime type name. All other types should be unaffected
			name := strings.ReplaceAll(c.symbolName(typ.Obj().Pkg(), typ.Obj().Name()), "runtime._", "")

			// Create a named struct with the same body as the underlying struct type
			result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)

			// Note: Need to cache structs right away to prevent a stack overflow
			// with a member type is or contains this struct type for any reason.
			c.types[typ] = result

			// Create the underlying struct type
			st := c.createType(ctx, structType)

			// Set the struct body to that of the underlying struct type
			llvm.StructSetBody(result.valueType, llvm.GetStructElementTypes(st.valueType), false)
		} else {
			result = c.createType(ctx, typ.Underlying())
		}
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.ArrayType(elementType.valueType, uint(typ.Len()))
	case *types.Interface:
		tt := c.createRuntimeType(ctx, "_interface")
		result.valueType = tt.valueType
		result.debugType = c.createDebugType(ctx, tt.spec)
		tt.spec = typ
	case *types.Pointer:
		elementType := c.createType(ctx, typ.Elem())
		// NOTE: This pointer is opaque!!!
		if elementType.valueType == nil {
			c.createType(ctx, typ.Elem())
			panic("element type is nil")
		}
		result.valueType = llvm.PointerType(elementType.valueType, 0)
	case *types.Chan:
		tt := c.createRuntimeType(ctx, "_channel")
		result.valueType = tt.valueType
		result.debugType = c.createDebugType(ctx, tt.spec)
	case *types.Signature:
		// Create this function type if it does not already exist
		fnType := c.createFunctionType(ctx, typ, false)
		result.valueType = llvm.PointerType(fnType, 0)
	case *types.Tuple:
		var memberTypes []llvm.LLVMTypeRef
		if typ != nil {
			for i := 0; i < typ.Len(); i++ {
				memberTypes = append(memberTypes, c.createType(ctx, typ.At(i).Type()).valueType)
			}
		}
		result.valueType = llvm.StructTypeInContext(c.currentContext(ctx), memberTypes, false)
	case *types.TypeParam:
		result = c.createType(ctx, typ.Constraint())
	default:
		panic("encountered unknown type")
	}

	// A known type should be created under all circumstances
	if result.valueType == nil {
		panic("no type was created")
	}

	// Cache the type for faster lookup later
	c.types[typ] = result

	// Return the type
	return result
}

func (c *Compiler) createRuntimeType(ctx context.Context, typename string) *Type {
	// Get the runtime package
	runtimePkg := c.currentPackage(ctx).Prog.ImportedPackage("runtime")
	if runtimePkg == nil {
		panic("no runtime package")
	}

	// Get the type within the runtime package
	t := runtimePkg.Type(typename)
	if t == nil {
		panic(typename + " does not exist in runtime")
	}

	return c.createType(ctx, t.Type())
}

func (c *Compiler) int1Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int1TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int8Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int8TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int16Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int16TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int32Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int32TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int64Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int64TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) boolType(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int1TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) float32(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.FloatTypeInContext(c.currentContext(ctx))
}

func (c *Compiler) float64(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.DoubleTypeInContext(c.currentContext(ctx))
}
