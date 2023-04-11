package compiler

import (
	"context"
	"go/types"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createType(ctx context.Context, typ types.Type) Type {
	// Check if this type was already created
	if result, ok := c.types[typ]; ok {
		return result
	}

	// Is this a named type?
	name := ""
	if named, ok := typ.(*types.Named); ok {
		// Extract the name
		name = named.Obj().Name()

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
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_unsigned, 0)
			}
		case types.Int8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_signed, 0)
			}
		case types.Uint16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_unsigned, 0)
			}
		case types.Int16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_signed, 0)
			}
		case types.Int, types.Int32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_unsigned, 0)
			}
		case types.Uint, types.Uint32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_signed, 0)
			}
		case types.Uint64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_unsigned, 0)
			}
		case types.Int64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_signed, 0)
			}
		case types.Uintptr:
			return c.uintptrType
		case types.UnsafePointer:
			result = c.uintptrType
		case types.Float32:
			result.valueType = llvm.FloatTypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_float, 0)
			}
		case types.Float64:
			result.valueType = llvm.DoubleTypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_float, 0)
			}
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String:
			result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), "string")
			structMembers := []llvm.LLVMTypeRef{
				c.uintptrType.valueType,                        // Ptr
				llvm.Int32TypeInContext(c.currentContext(ctx)), // Len
			}
			llvm.StructSetBody(result.valueType, structMembers, false)

			if c.GenerateDebugInfo {
				// Create the debug information for the underlying struct
				result.debugType = llvm.DIBuilderCreateStructType(
					c.dibuilder,
					c.currentScope(ctx),
					"string",
					nil,
					0,
					4, // Alignment - TODO: Get this information from the target information
					0, 0, nil, nil, 0, nil, "")
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					"array",
					nil,
					0,
					4,
					4, // Alignment - TODO: Get this information from the target information
					0, // Offset
					0, // Flags
					c.uintptrType.debugType)
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					"len",
					nil,
					0,
					4,
					4,  // Alignment - TODO: Get this information from the target information
					32, // Offset - TODO: Use the value above to calculate the offset within the struct
					0,  // Flags
					llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
			}
		default:
			panic("encountered unknown basic type")
		}
	case *types.Struct:
		if c.GenerateDebugInfo {
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

			if c.GenerateDebugInfo {
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					field.Name(),
					nil,
					0,
					4,
					4,      // Alignment - TODO: Get this information from the target information
					offset, // Offset
					0,      // Flags
					fieldType.debugType)

				offset += llvm.DITypeGetSizeInBits(fieldType.debugType)
			}
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)
	case *types.Map:
		panic("Not implemented")
	case *types.Slice:
		structMembers := []llvm.LLVMTypeRef{
			c.uintptrType.valueType,                        // Ptr
			llvm.Int32TypeInContext(c.currentContext(ctx)), // Len
			llvm.Int32TypeInContext(c.currentContext(ctx)), // Cap
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)

		if c.GenerateDebugInfo {
			// Create the debug information for the underlying struct
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"array",
				nil,
				0,
				4,
				4, // Alignment - TODO: Get this information from the target information
				0, // Offset
				0, // Flags
				c.uintptrType.debugType)
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"len",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				32, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"cap",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
		}
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.ArrayType2(elementType.valueType, uint64(typ.Len()))
		if c.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateArrayType(
				c.dibuilder,
				uint64(typ.Len()),
				4, // TODO: Get this information for the target machine descriptor
				elementType.debugType,
				nil)
		}
	case *types.Interface:
		structMembers := []llvm.LLVMTypeRef{
			c.uintptrType.valueType, // Ptr
			c.uintptrType.valueType, // Vtable
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)

		if c.GenerateDebugInfo {
			// Create the debug information for the underlying struct
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")

			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"ptr",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				c.uintptrType.debugType)

			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"vtable",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				c.uintptrType.debugType)
		}
	case *types.Pointer:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.PointerType(elementType.valueType, 0)
		if c.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateObjectPointerType(c.dibuilder, elementType.debugType)
		}
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
