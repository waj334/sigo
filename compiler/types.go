package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createType(ctx context.Context, typ types.Type) Type {
	// Check if this type was already created
	if result, ok := c.types[typ]; ok {
		c.printf(Debug, "Returning type from cache: %s\n", typ.String())
		return result
	}

	typename := typ.String()
	c.printf(Debug, "Creating type: %s\n", typename)

	// Keep a copy of the original type since typ will be reassigned if it is a
	// named type.
	keyType := typ

	// Is this a named type?
	name := ""
	if named, ok := typ.(*types.Named); ok {
		// Extract the name
		name = named.Obj().Name()

		if pkg := named.Obj().Pkg(); pkg != nil {
			// Prepend the package path
			name = pkg.Path() + "." + name
		}

		// Create the underlying type
		typ = named.Underlying()
	}

	var result Type

	// Create the equivalent LLVM type
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_unsigned, 0)
			}
		case types.Int8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_signed, 0)
			}
		case types.Uint16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_unsigned, 0)
			}
		case types.Int16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_signed, 0)
			}
		case types.Int, types.Int32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_unsigned, 0)
			}
		case types.Uint, types.Uint32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_signed, 0)
			}
		case types.Uint64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_unsigned, 0)
			}
		case types.Int64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_signed, 0)
			}
		case types.Uintptr:
			return c.uintptrType
		case types.UnsafePointer:
			result = c.ptrType
		case types.Float32:
			result.valueType = llvm.FloatTypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_float, 0)
			}
		case types.Float64:
			result.valueType = llvm.DoubleTypeInContext(c.currentContext(ctx))
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_float, 0)
			}
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String:
			runtimeType, ok := c.findRuntimeType(ctx, "runtime/internal/go.stringDescriptor")
			if ok {
				result = runtimeType
			} else {
				panic("runtime type does not exist")
			}
		case types.Bool:
			result.valueType = llvm.Int1Type()
			if c.options.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_boolean, 0)
			}
		default:
			panic("encountered unknown basic type")
		}
	case *types.Struct:
		if c.options.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")
		}

		var structMembers []llvm.LLVMTypeRef
		offset := uint64(0)
		for i := 0; i < typ.NumFields(); i++ {
			field := typ.Field(i)
			fieldType := c.createType(ctx, field.Type())
			structMembers = append(structMembers, fieldType.valueType)

			if c.options.GenerateDebugInfo {
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					field.Name(),
					nil,
					0,
					4,
					4,        // Alignment - TODO: Get this information from the target information
					offset*8, // Offset
					0,        // Flags
					fieldType.debugType)

				offset += llvm.StoreSizeOfType(c.options.Target.dataLayout, fieldType.valueType)
			}
		}
		result.valueType = llvm.StructTypeInContext(c.currentContext(ctx), structMembers, false)
	case *types.Map:
		runtimeType, ok := c.findRuntimeType(ctx, "runtime/internal/go.mapDescriptor")
		if ok {
			result = runtimeType
		} else {
			panic("runtime type does not exist")
		}
	case *types.Slice:
		runtimeType, ok := c.findRuntimeType(ctx, "runtime/internal/go.sliceDescriptor")
		if ok {
			result = runtimeType
		} else {
			panic("runtime type does not exist")
		}
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.ArrayType2(elementType.valueType, uint64(typ.Len()))
		if c.options.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateArrayType(
				c.dibuilder,
				uint64(typ.Len()),
				4, // TODO: Get this information for the target machine descriptor
				elementType.debugType,
				nil)
		}
	case *types.Interface:
		runtimeType, ok := c.findRuntimeType(ctx, "runtime/internal/go.interfaceDescriptor")
		if ok {
			result.valueType = runtimeType.valueType

		} else {
			panic("runtime type does not exist")
		}
	case *types.Pointer:
		elementType := c.createType(ctx, typ.Elem())
		// NOTE: This pointer is opaque!!!
		result.valueType = llvm.PointerType(elementType.valueType, 0)

		if c.options.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateObjectPointerType(c.dibuilder, elementType.debugType)
		}
	case *types.Chan:
		runtimeType, ok := c.findRuntimeType(ctx, "runtime/internal/go.channelDescriptor")
		if ok {
			result = runtimeType
		} else {
			panic("runtime type does not exist")
		}
	default:
		panic("encountered unknown type")
	}

	// A known type should be created under all circumstances
	if result.valueType == nil {
		panic("no type was created")
	}

	// Store the type's name
	result.name = name

	// Store the go type
	result.spec = typ

	// Create the type descriptor
	//result.descriptor = c.createTypeDescriptor(ctx, result)

	// Cache the type for faster lookup later
	c.types[keyType] = result

	// Return the type
	return result
}

func (c *Compiler) findRuntimeType(ctx context.Context, typename string) (Type, bool) {
	// Search for the type by name
	if len(typename) > 0 {
		for _, t := range c.types {
			if t.name == typename {
				return t, true
			}
		}
	}

	// Type not found
	return Type{}, false
}
