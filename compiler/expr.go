package compiler

import (
	"context"
	"go/constant"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createExpression(ctx context.Context, expr ssa.Value) (value llvm.LLVMValueRef, err error) {
	c.printf(Debug, "Processing expression %T: %s\n", expr, expr.String())

	// Check if this value was cached
	if value, ok := c.values[expr]; ok {
		// This value might be a variable. Get the value from the variable's (alloca) address
		//if _, ok := c.variables[value]; ok {
		//	// Ignore *ssa.Alloc since these should be pointers directly
		//	if _, ok := expr.(*ssa.Alloc); !ok {
		//		value = c.getVariableValue(value)
		//		c.println(Debug, "Loading value from variable")
		//		return value, nil
		//	}
		//}

		// Return the cached value
		c.println(Debug, "returning cached value")
		return value, nil
	}

	// Evaluate the expression
	switch expr := expr.(type) {
	case *ssa.Alloc:
		// Get the type of which memory will be allocated for
		typ := c.createType(ctx, expr.Type().Underlying().(*types.Pointer).Elem())

		// NOTE: Some stack allocations will be moved to the heap later if they
		//       are too big for the stack.
		// Get the size of the type
		size := llvm.StoreSizeOfType(c.options.Target.dataLayout, typ.valueType)

		if expr.Heap {
			// Heap allocations will store the address of the runtime allocation in
			// the alloca. Allocate space for this pointer on the stack.
			//typ = c.createType(ctx, expr.Type().Underlying())

			// Create the alloca to hold the address on the stack
			value = llvm.BuildAlloca(c.builder, c.ptrType.valueType, expr.Comment)
			value = c.createVariable(ctx, expr.Comment, value, expr.Type().Underlying().(*types.Pointer))

			// Create the runtime call to allocate some memory on the heap
			addr, err := c.createRuntimeCall(ctx, "alloc", []llvm.LLVMValueRef{llvm.ConstInt(c.uintptrType.valueType, size, false)})
			if err != nil {
				return nil, err
			}

			// Store the address at the alloc
			llvm.BuildStore(c.builder, addr, value)
		} else {
			// Create an alloca to hold the value on the stack
			value = llvm.BuildAlloca(c.builder, typ.valueType, expr.Comment)
			value = c.createVariable(ctx, expr.Comment, value, expr.Type().Underlying().(*types.Pointer).Elem())

			// Zero-initialize the stack variable
			if size > 0 {
				llvm.BuildStore(c.builder, llvm.ConstNull(typ.valueType), value)
			}
		}

		// Finally create a variable to hold debug information about this alloca
		//value = c.createVariable(ctx, expr.Name(), value, typ)
	case *ssa.BinOp:
		value, err = c.createBinOp(ctx, expr)
	case *ssa.Call:
		switch callExpr := expr.Common().Value.(type) {
		case *ssa.Builtin:
			value, err = c.createBuiltinCall(ctx, callExpr, expr.Call.Args)
		case *ssa.Function:
			value, err = c.createFunctionCall(ctx, callExpr, expr.Call.Args)
		case *ssa.MakeClosure:
			// a *MakeClosure, indicating an immediately applied function
			// literal with free variables.
			panic("not implemented")
		default:
			// any other value, indicating a dynamically dispatched function
			// call.
			panic("not implemented")
		}
	case *ssa.Const:
		constType := c.createType(ctx, expr.Type())
		if expr.Value == nil {
			value = llvm.ConstNull(constType.valueType)
		} else {
			switch expr.Value.Kind() {
			case constant.Bool:
				if constant.BoolVal(expr.Value) {
					value = llvm.ConstInt(constType.valueType, 1, false)
				} else {
					value = llvm.ConstInt(constType.valueType, 0, false)
				}
			case constant.String:
				strValue := constant.StringVal(expr.Value)
				value = c.createGlobalString(ctx, strValue)
			case constant.Int:
				constVal, _ := constant.Int64Val(expr.Value)
				value = llvm.ConstInt(constType.valueType, uint64(constVal), false)
			case constant.Float:
				constVal, _ := constant.Float64Val(expr.Value)
				value = llvm.ConstReal(constType.valueType, constVal)
			case constant.Complex:
				panic("not implemented")
			default:
				panic("unknown default value")
			}
		}
	case *ssa.Convert:
		typeFrom := c.createType(ctx, expr.X.Type())
		typeTo := c.createType(ctx, expr.Type())

		fromSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, typeFrom.valueType)
		toSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, typeTo.valueType)

		fromValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		switch typeX := expr.X.Type().Underlying().(type) {
		case *types.Basic:
			switch otherType := expr.Type().(type) {
			case *types.Basic:
				fromIsInteger := typeX.Info()&types.IsInteger != 0
				toIsInteger := otherType.Info()&types.IsInteger != 0

				fromIsFloat := typeX.Info()&types.IsFloat != 0
				toIsFloat := otherType.Info()&types.IsFloat != 0

				fromIsComplex := typeX.Info()&types.IsComplex != 0
				toIsComplex := otherType.Info()&types.IsComplex != 0

				fromIsUnsigned := typeX.Info()&types.IsUnsigned != 0
				toIsUnsigned := otherType.Info()&types.IsUnsigned != 0

				fromIsString := typeX.Info()&types.IsString != 0
				toIsString := otherType.Info()&types.IsString != 0

				switch typeX.Kind() {
				case types.UnsafePointer:
					if otherType.Kind() == types.Uintptr {
						value = llvm.BuildPtrToInt(c.builder, fromValue, typeTo.valueType, "")
					} else {
						value = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
					}
				case types.Uintptr:
					if otherType.Kind() == types.UnsafePointer {
						value = llvm.BuildIntToPtr(c.builder, fromValue, typeTo.valueType, "")
						break
					}
					fallthrough
				default:
					if fromIsInteger && toIsInteger {
						if fromSize > toSize {
							value = llvm.BuildTrunc(c.builder, fromValue, typeTo.valueType, "")
						} else if fromIsUnsigned && toIsUnsigned {
							value = llvm.BuildZExt(c.builder, fromValue, typeTo.valueType, "")
						} else if !fromIsUnsigned && !toIsUnsigned {
							value = llvm.BuildSExt(c.builder, fromValue, typeTo.valueType, "")
						} else {
							// Signed to Unsigned or vice versa doesn't require an
							// explicit instruction since the underlying bit format is the same.
							value = fromValue
						}
					} else if fromIsFloat && toIsFloat {
						if fromSize > toSize {
							value = llvm.BuildFPTrunc(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value = llvm.BuildFPExt(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsFloat && toIsInteger {
						if toIsUnsigned {
							value = llvm.BuildFPToSI(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value = llvm.BuildFPToUI(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsInteger && toIsFloat {
						if toIsUnsigned {
							value = llvm.BuildUIToFP(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value = llvm.BuildSIToFP(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsComplex && toIsComplex {
						panic("not implemented")
					} else if fromIsString {
						otherSliceType := otherType.Underlying().(*types.Slice)
						otherElementType := otherSliceType.Elem().(*types.Basic)
						if otherElementType.Kind() == types.Byte {
							//TODO: Create runtime call for this
							panic("not implemented")
						} else if otherElementType.Kind() == types.Rune {
							//TODO: Create runtime call for this
							panic("not implemented")
						} else {
							panic("not implemented")
						}
					} else if fromIsInteger && toIsString {
						panic("not implemented")
					}
				}
			case *types.Pointer:
				value = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
			}
		case *types.Pointer:
			otherType := expr.Type().(*types.Basic)
			if otherType.Kind() == types.UnsafePointer {
				value = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
			} else {
				panic("not implemented")
			}
		case *types.Slice:
			panic("not implemented")
		}
	case *ssa.Parameter:
		if fn, ok := c.functions[expr.Parent().Signature]; ok {
			// Locate the parameter in the function
			for i, param := range expr.Parent().Params {
				if param == expr {
					value = llvm.GetParam(fn.value, uint(i))
				}
			}
		} else {
			panic("function does not exist")
		}

		// All parameters should be allocated on the stack.
		/*value = llvm.BuildAlloca(c.builder, typ.valueType, expr.Name())

		// Finally create a variable to hold debug information about this alloca
		//value = c.createVariable(ctx, expr.Name(), alloca, typ)*/
	case *ssa.ChangeInterface:
		panic("not implemented")
	case *ssa.ChangeType:
		panic("not implemented")
	case *ssa.Extract:
		// Get the return struct (tuple)
		structValue, err := c.createExpression(ctx, expr.Tuple)
		if err != nil {
			return nil, err
		}

		structType := llvm.TypeOf(structValue)

		// Get the address of the field within the return struct (tuple)
		fieldType, addr := c.structFieldAddress(structValue, structType, expr.Index)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.Field:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		structType := llvm.TypeOf(structValue)

		// Get the address of the field within the struct
		fieldType, addr := c.structFieldAddress(structValue, structType, expr.Field)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.FieldAddr:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		var structType llvm.LLVMTypeRef

		if pointerType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
			structType = c.createType(ctx, pointerType.Elem()).valueType
		} else {
			structType = llvm.TypeOf(structValue)
		}

		//Return the address
		_, value = c.structFieldAddress(structValue, structType, expr.Field)
	case *ssa.Global:
		// Create a global value
		globalType := c.createType(ctx, expr.Type())

		isExternal := false
		isExported := false
		info, ok := c.options.Symbols[expr.Object().Id()]
		if ok {
			isExternal = info.ExternalLinkage
			isExported = info.Exported
		}

		// Cannot be both exported and external
		if isExported && isExternal {
			panic("global cannot be both external and exported")
		}

		if isExternal {
			// Create a global with external linkage to some variable with the specified link name.
			value = llvm.AddGlobal(c.module, globalType.valueType, info.LinkName)
			llvm.SetLinkage(value, llvm.LLVMLinkage(llvm.ExternalLinkage))
		} else {
			value = c.createGlobalValue(ctx, llvm.ConstNull(globalType.valueType), types.Id(c.currentPackage(ctx).Pkg, expr.Name()))
			if !isExported {
				llvm.SetLinkage(value, llvm.LLVMLinkage(llvm.PrivateLinkage))
			}
		}
	case *ssa.Index:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return nil, err
		}

		elementType := llvm.GetElementType(llvm.TypeOf(arrayValue))

		// Get the address of the element at the index within the array
		addr := c.arrayElementAddress(ctx, arrayValue, elementType, indexValue)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, elementType, addr, "")
	case *ssa.IndexAddr:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return nil, err
		}

		// The container can be a slice or an array
		if sliceType, ok := expr.X.Type().Underlying().(*types.Slice); ok {
			// Get the element type of the slice
			elementType := c.createType(ctx, sliceType.Elem())

			// Get the element size of the slice
			elementSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType.valueType)

			// Create a runtime call to retrieve the address of the element at index I
			value, err = c.createRuntimeCall(ctx, "sliceIndex", []llvm.LLVMValueRef{
				c.addressOf(arrayValue),
				indexValue,
				llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), elementSize, false),
			})
			if err != nil {
				return nil, err
			}
		} else {
			var arrayType llvm.LLVMTypeRef
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				// Get the expected array type, so we can cast the pointer
				// value from below to this type.
				arrayType = c.createType(ctx, ptrType.Elem()).valueType

				// Bitcast the resulting value to a pointer of the array type
				arrayValue = llvm.BuildBitCast(c.builder, arrayValue, llvm.PointerType(arrayType, 0), "")
			} else {
				arrayType = llvm.TypeOf(arrayValue)
			}

			// Get the address of the element at the index within the array
			value = c.arrayElementAddress(ctx, arrayValue, llvm.GetElementType(arrayType), indexValue)
		}
	case *ssa.Lookup:
		panic("not implemented")
	case *ssa.MakeChan:
		panic("not implemented")
	case *ssa.MakeClosure:
		panic("not implemented")
	case *ssa.MakeInterface:
		value, err = c.makeInterface(ctx, expr.X)
		if err != nil {
			return nil, err
		}
	case *ssa.MakeSlice:
		panic("not implemented")
	case *ssa.MultiConvert:
		panic("not implemented")
	case *ssa.Next:
		panic("not implemented")
	case *ssa.Phi:
		phiType := c.createType(ctx, expr.Type())

		// Build the Phi node operator
		value = llvm.BuildPhi(c.builder, phiType.valueType, "")

		// Create a variable for this Phi node
		//value = c.createVariable(ctx, expr.Comment, phiValue, phiType)

		// Cache the PHI value now to prevent a stack overflow in the call to createValues below
		if _, ok := c.values[expr]; ok {
			panic("PHI node value already generated")
		}
		c.values[expr] = value

		// Create values for each of the Phi node's edges
		edgeValues, err := c.createValues(ctx, expr.Edges)
		if err != nil {
			return nil, err
		}

		// Get the blocks each edge value belongs to
		// NOTE: Edges[i] is value for Block().Preds[i]
		var blocks []llvm.LLVMBasicBlockRef
		for i, _ := range expr.Edges {
			block := c.blocks[expr.Block().Preds[i]]
			blocks = append(blocks, block)
		}

		// Add the edges
		llvm.AddIncoming(value, edgeValues, blocks)
	case *ssa.Slice:
		var array, low, high, max llvm.LLVMValueRef
		if expr.Low != nil {
			low, err = c.createExpression(ctx, expr.Low)
			if err != nil {
				return nil, err
			}
		}

		if expr.High != nil {
			high, err = c.createExpression(ctx, expr.High)
			if err != nil {
				return nil, err
			}
		}

		if expr.Max != nil {
			max, err = c.createExpression(ctx, expr.Max)
			if err != nil {
				return nil, err
			}
		}

		array, err = c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		var elementType llvm.LLVMTypeRef
		numElements := uint64(0)

		switch t := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			elementType = c.createType(ctx, t.Elem()).valueType
		case *types.Basic:
			elementType = llvm.Int8Type()
		case *types.Pointer:
			if tt, ok := t.Elem().Underlying().(*types.Array); ok {
				elementType = c.createType(ctx, tt.Elem()).valueType
				numElements = uint64(tt.Len())
			} else {
				panic("invalid pointer type")
			}
		}

		value = c.createSlice(ctx, array, elementType, numElements, low, high, max)
	case *ssa.SliceToArrayPointer:
		panic("not implemented")
	case *ssa.TypeAssert:
		x, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		newType := c.createType(ctx, expr.AssertedType)

		// The value needs to be passed by address. Create an alloca for it and store the value
		xAlloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(x), "")
		llvm.BuildStore(c.builder, x, xAlloca)

		xAlloca = llvm.BuildBitCast(c.builder, xAlloca, c.ptrType.valueType, "")

		// create the runtime call
		result, err := c.createRuntimeCall(ctx, "typeAssert", []llvm.LLVMValueRef{
			xAlloca,
			c.createTypeDescriptor(ctx, c.createType(ctx, expr.X.Type())),
			c.createTypeDescriptor(ctx, newType),
		})
		if err != nil {
			return nil, err
		}

		// Extract values from return
		// Create a pointer to the struct
		alloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(result), "")
		llvm.BuildStore(c.builder, result, alloca)
		_, newObj := c.structFieldAddress(alloca, llvm.TypeOf(result), 0)
		_, isOk := c.structFieldAddress(alloca, llvm.TypeOf(result), 1)

		// TODO: There definitely more involved than this. Gonna try to
		//       implement the semantics in Go code rather than hardcode it
		//       here. I should be able to load the resulting object from
		//       the address obtained from the runtime call.
		obj := llvm.BuildLoad2(c.builder, newType.valueType, newObj, "")

		if expr.CommaOk {
			// Return the obj and the status
			value = llvm.ConstStruct([]llvm.LLVMValueRef{
				llvm.GetUndef(newType.valueType),
				llvm.GetUndef(llvm.Int1Type()),
			}, false)
			value = llvm.BuildInsertValue(c.builder, value, obj, uint(0), "")
			value = llvm.BuildInsertValue(c.builder, value, isOk, uint(1), "")
		} else {

			value = obj
		}
	case *ssa.UnOp:
		value, err = c.createUpOp(ctx, expr)
	}

	if value == nil {
		panic("nil value")
	}

	// Cache the value
	c.values[expr] = value

	return value, nil
	//return c.getVariableValue(value), nil
}
