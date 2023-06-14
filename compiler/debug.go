package compiler

import (
	"context"
	"go/types"
	"omibyte.io/sigo/llvm"
)

type test struct {
	t *test
}

func (c *Compiler) mapPrimitiveName(in string) string {
	if c.options.PrimitivesAsCTypes {
		switch in {
		case "uint8", "byte":
			return "unsigned char"
		case "int8":
			return "char"
		case "uint16":
			return "unsigned short"
		case "int16":
			return "short"
		case "uint32":
			return "unsigned int"
		case "int32":
			return "int"
		case "uint64":
			return "unsigned long long int"
		case "int64":
			return "long long int"
		case "float32":
			return "float"
		case "float64":
			return "double"
		case "unsafe.Pointer":
			return "void*"
		case "uintptr":
			return "unsigned long"
		default:
			return in
		}
	}
	return in
}

func (c *Compiler) createDebugType(ctx context.Context, typ types.Type) (ditype llvm.LLVMMetadataRef) {
	// Check if this type was already created
	t := c.createType(ctx, typ)
	if t.debugType != nil {
		c.printf(Debug, "Returning debug info from cache: %s\n", typ.String())
		return t.debugType
	}

	c.printf(Debug, "Creating debug info for type: %s\n", typ.String())
	defer c.printf(Debug, "Done creating debug info for type: %s\n", typ.String())

	// Create the equivalent LLVM type
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 8, DW_ATE_unsigned, 0)
		case types.Int8:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 8, DW_ATE_signed, 0)
		case types.Uint16:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 16, DW_ATE_unsigned, 0)
		case types.Int16:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 16, DW_ATE_signed, 0)
		case types.Int, types.Int32:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 4, DW_ATE_unsigned, 0)
		case types.Uint, types.Uint32:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 32, DW_ATE_signed, 0)
		case types.Uint64:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 64, DW_ATE_unsigned, 0)
		case types.Int64:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 64, DW_ATE_signed, 0)
		case types.Uintptr:
			ditype = c.uintptrType.debugType
		case types.UnsafePointer:
			ditype = c.ptrType.debugType
		case types.Float32:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 32, DW_ATE_float, 0)
		case types.Float64:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, c.mapPrimitiveName(typ.Name()), 64, DW_ATE_float, 0)
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String:
			ditype = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				nil,
				"string",
				nil,
				0,
				llvm.SizeOfTypeInBits(c.options.Target.dataLayout, t.valueType),
				llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
				0, nil, []llvm.LLVMMetadataRef{
					llvm.DIBuilderCreateMemberType(
						c.dibuilder,
						nil,
						"array",
						nil,
						0,
						llvm.SizeOfTypeInBits(c.options.Target.dataLayout, c.ptrType.valueType),
						llvm.ABIAlignmentOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
						llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 0)*8,
						0,
						c.ptrType.debugType),
					llvm.DIBuilderCreateMemberType(
						c.dibuilder,
						nil,
						"len",
						nil,
						0,
						llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))),
						llvm.ABIAlignmentOfType(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx)))*8,
						llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 1)*8,
						0,
						llvm.DIBuilderCreateBasicType(c.dibuilder, "int",
							llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))), DW_ATE_unsigned, 0)),
				}, 0, nil, "")
		case types.Bool:
			ditype = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_boolean, 0)
		default:
			panic("encountered unknown basic type")
		}
	case *types.Struct:
		// Forward declare the debug type
		// Note: Need to cache structs right away to prevent a stack overflow
		// with a member type is or contains this struct type for any reason.
		t.debugType = llvm.TemporaryMDNode(c.currentContext(ctx), []llvm.LLVMMetadataRef{})

		if llvm.GetTypeKind(t.valueType) != llvm.StructTypeKind {
			println(llvm.PrintTypeToString(t.valueType))
			panic("value is not a struct")
		}

		var structDITypes []llvm.LLVMMetadataRef
		for i := 0; i < typ.NumFields(); i++ {
			field := typ.Field(i)
			fieldType := c.createType(ctx, field.Type())
			structDITypes = append(structDITypes, llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				nil,
				field.Name(),
				nil,
				0,
				llvm.SizeOfTypeInBits(c.options.Target.dataLayout, fieldType.valueType),
				llvm.ABIAlignmentOfType(c.options.Target.dataLayout, fieldType.valueType)*8, // Alignment
				llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, uint(i))*8,   // Offset
				0, // Flags
				c.createDebugType(ctx, field.Type())))
		}

		ditype = llvm.DIBuilderCreateStructType(
			c.dibuilder,
			nil,
			"",
			nil,
			0,
			llvm.StoreSizeOfType(c.options.Target.dataLayout, t.valueType)*8,
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8, // Alignment
			0, nil, structDITypes, 0, nil, typ.String())

		// Replace all uses of the temp type created earlier
		llvm.MetadataReplaceAllUsesWith(t.debugType, ditype)
	case *types.Map:
		ditype = llvm.DIBuilderCreateStructType(
			c.dibuilder,
			nil,
			"map",
			nil,
			0,
			llvm.SizeOfTypeInBits(c.options.Target.dataLayout, t.valueType),
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
			0, nil, nil, 0, nil, "")
	case *types.Named:
		ditype = llvm.DIBuilderCreateTypedef(
			c.dibuilder,
			c.createDebugType(ctx, typ.Underlying()),
			typ.Obj().Name(),
			nil,
			0,
			nil,
			0,
		)
	case *types.Slice:
		ditype = llvm.DIBuilderCreateStructType(
			c.dibuilder,
			nil,
			"slice",
			nil,
			0,
			llvm.SizeOfTypeInBits(c.options.Target.dataLayout, t.valueType),
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
			0, nil, []llvm.LLVMMetadataRef{
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					nil,
					"array",
					nil,
					0,
					llvm.SizeOfTypeInBits(c.options.Target.dataLayout, c.ptrType.valueType),
					llvm.ABIAlignmentOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
					llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 0)*8,
					0,
					c.ptrType.debugType),
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					nil,
					"len",
					nil,
					0,
					llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))),
					llvm.ABIAlignmentOfType(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx)))*8,
					llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 1)*8,
					0,
					llvm.DIBuilderCreateBasicType(c.dibuilder, "int",
						llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))), DW_ATE_unsigned, 0)),
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					nil,
					"cap",
					nil,
					0,
					llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))),
					llvm.ABIAlignmentOfType(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx)))*8,
					llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 2)*8,
					0,
					llvm.DIBuilderCreateBasicType(c.dibuilder, "int",
						llvm.SizeOfTypeInBits(c.options.Target.dataLayout, llvm.Int32TypeInContext(c.currentContext(ctx))), DW_ATE_unsigned, 0)),
			}, 0, nil, "")
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		ditype = llvm.DIBuilderCreateArrayType(
			c.dibuilder,
			uint64(typ.Len())*llvm.SizeOfTypeInBits(c.options.Target.dataLayout, elementType.valueType),
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
			c.createDebugType(ctx, typ.Elem()),
			[]llvm.LLVMMetadataRef{
				llvm.DIBuilderGetOrCreateSubrange(c.dibuilder, 0, typ.Len()),
			})
	case *types.Interface:
		ditype = llvm.DIBuilderCreateStructType(
			c.dibuilder,
			nil,
			"interface",
			nil,
			0,
			llvm.SizeOfTypeInBits(c.options.Target.dataLayout, t.valueType),
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
			0, nil, []llvm.LLVMMetadataRef{
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					nil,
					"info",
					nil,
					0,
					llvm.SizeOfTypeInBits(c.options.Target.dataLayout, c.ptrType.valueType),
					llvm.ABIAlignmentOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
					llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 0)*8,
					0,
					c.ptrType.debugType),
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					nil,
					"ptr",
					nil,
					0,
					llvm.SizeOfTypeInBits(c.options.Target.dataLayout, c.ptrType.valueType),
					llvm.ABIAlignmentOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
					llvm.OffsetOfElement(c.options.Target.dataLayout, t.valueType, 1)*8,
					0,
					c.ptrType.debugType),
			}, 0, nil, "")
	case *types.Pointer:
		// Create a temporary type for this pointer
		t.debugType = llvm.TemporaryMDNode(c.currentContext(ctx), []llvm.LLVMMetadataRef{})

		// Create the element debug info
		elementDbg := c.createDebugType(ctx, typ.Elem())

		// Create the pointer type
		ditype = llvm.DIBuilderCreatePointerType(
			c.dibuilder,
			elementDbg,
			llvm.StoreSizeOfType(c.options.Target.dataLayout, t.valueType)*8,
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8, 0, "")

		// Replace the temp type
		llvm.MetadataReplaceAllUsesWith(t.debugType, ditype)
	case *types.Chan:
		ditype = llvm.DIBuilderCreateStructType(
			c.dibuilder,
			nil,
			"channel",
			nil,
			0,
			llvm.SizeOfTypeInBits(c.options.Target.dataLayout, t.valueType),
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, t.valueType)*8,
			0, nil, nil, 0, nil, "")
	case *types.Signature:
		// Create the debug information for this function
		var argDiTypes []llvm.LLVMMetadataRef
		for i := 0; i < typ.Params().Len(); i++ {
			arg := typ.Params().At(i)
			argDiTypes = append(argDiTypes, c.createDebugType(ctx, arg.Type()))
		}

		ditype = llvm.DIBuilderCreateSubroutineType(
			c.dibuilder,
			nil,
			argDiTypes,
			0)
	case *types.Tuple:
		panic("cannot create debug information for this type")
	default:
		panic("encountered unknown type")
	}

	// Cache the type for faster lookup later
	t.debugType = ditype

	return
}

func (c *Compiler) createVariable(ctx context.Context, name string, value Value, valueType types.Type) Value {
	c.printf(Debug, "Creating variable for %s (%s)\n", name, valueType.String())
	defer c.printf(Debug, "Done creating variable for %s (%s)\n", name, valueType.String())

	dbgType := c.createDebugType(ctx, valueType)
	if _, ok := valueType.(*types.Signature); ok {
		dbgType = llvm.DIBuilderCreatePointerType(
			c.dibuilder,
			dbgType,
			llvm.StoreSizeOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
			llvm.ABIAlignmentOfType(c.options.Target.dataLayout, c.ptrType.valueType)*8,
			0, "")
	}

	scope, _ := c.instructionScope(value.spec)
	if scope == nil {
		scope = c.compileUnit
	}

	// Create the debug information about the variable
	value.dbg = llvm.DIBuilderCreateAutoVariable(
		c.dibuilder,
		scope,
		name,
		value.DebugFile(),
		uint(value.Pos().Line),
		dbgType,
		true,
		0,
		0)

	var expression llvm.LLVMMetadataRef
	if value.heap {
		// Have the debugger dereference the object pointer to obtain the underlying object
		ops := []uint64{
			uint64(DW_OP_deref),
		}
		expression = llvm.DIBuilderCreateExpression(c.dibuilder, ops)
	} else {
		expression = llvm.DIBuilderCreateExpression(c.dibuilder, nil)
	}

	// Add debug info about the declaration
	llvm.DIBuilderInsertDeclareAtEnd(
		c.dibuilder,
		value,
		value.dbg,
		expression,
		value.DebugPos(ctx),
		c.currentEntryBlock(ctx))

	return value
}
