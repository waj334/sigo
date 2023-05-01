package compiler

import (
	"context"
	"go/constant"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createExpression(ctx context.Context, expr ssa.Value) (value Value, err error) {
	c.printf(Debug, "Processing expression %T: %s\n", expr, expr.String())

	// Check if this value was cached
	if value, ok := c.values[expr]; ok {
		// Return the cached value
		c.println(Debug, "returning cached value")
		return value, nil
	}

	// initialize the value
	value.cc = c

	// Evaluate the expression
	switch expr := expr.(type) {
	case *ssa.Alloc:
		// Get the type of which memory will be allocated for
		elementType := c.createType(ctx, expr.Type().Underlying().(*types.Pointer).Elem())

		// NOTE: Some stack allocations will be moved to the heap later if they
		//       are too big for the stack.
		// Get the size of the type
		size := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType.valueType)

		if expr.Heap {
			// Create the alloca to hold the address on the stack
			value.LLVMValueRef = llvm.BuildAlloca(c.builder, c.ptrType.valueType, expr.Comment)
			c.createVariable(ctx, expr.Comment, value, expr.Type().Underlying())

			// Mark this value as one that is on the heap
			value.heap = true

			// Create the runtime call to allocate some memory on the heap
			addr, err := c.createRuntimeCall(ctx, "alloc",
				[]llvm.LLVMValueRef{llvm.ConstInt(c.uintptrType.valueType, size, false)})
			if err != nil {
				return invalidValue, err
			}

			// Store the address at the alloc
			llvm.BuildStore(c.builder, addr, value.LLVMValueRef)
		} else {
			// Create an alloca to hold the value on the stack
			value.LLVMValueRef = llvm.BuildAlloca(c.builder, elementType.valueType, expr.Comment)
			c.createVariable(ctx, expr.Comment, value, elementType.spec)

			// Zero-initialize the stack variable
			if size > 0 {
				llvm.BuildStore(c.builder, llvm.ConstNull(elementType.valueType), value.LLVMValueRef)
			}
		}

		// Finally create a variable to hold debug information about this alloca
		//value = c.createVariable(ctx, expr.Name(), value, typ)
	case *ssa.BinOp:
		value.LLVMValueRef, err = c.createBinOp(ctx, expr)
	case *ssa.Call:
		switch callExpr := expr.Common().Value.(type) {
		case *ssa.Builtin:
			value.LLVMValueRef, err = c.createBuiltinCall(ctx, callExpr, expr.Call.Args)
		case *ssa.Function:
			value.LLVMValueRef, err = c.createFunctionCall(ctx, callExpr, expr.Call.Args)
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
			value.LLVMValueRef = llvm.ConstNull(constType.valueType)
		} else {
			switch expr.Value.Kind() {
			case constant.Bool:
				if constant.BoolVal(expr.Value) {
					value.LLVMValueRef = llvm.ConstInt(constType.valueType, 1, false)
				} else {
					value.LLVMValueRef = llvm.ConstInt(constType.valueType, 0, false)
				}
			case constant.String:
				strValue := constant.StringVal(expr.Value)
				value.LLVMValueRef = c.createConstantString(ctx, strValue)
			case constant.Int:
				constVal, _ := constant.Int64Val(expr.Value)
				value.LLVMValueRef = llvm.ConstInt(constType.valueType, uint64(constVal), false)
			case constant.Float:
				constVal, _ := constant.Float64Val(expr.Value)
				value.LLVMValueRef = llvm.ConstReal(constType.valueType, constVal)
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
			return invalidValue, err
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
						value.LLVMValueRef = llvm.BuildPtrToInt(c.builder, fromValue, typeTo.valueType, "")
					} else {
						value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
					}
				case types.Uintptr:
					if otherType.Kind() == types.UnsafePointer {
						value.LLVMValueRef = llvm.BuildIntToPtr(c.builder, fromValue, typeTo.valueType, "")
						break
					}
					fallthrough
				default:
					if fromIsInteger && toIsInteger {
						if fromSize > toSize {
							value.LLVMValueRef = llvm.BuildTrunc(c.builder, fromValue, typeTo.valueType, "")
						} else if fromIsUnsigned && toIsUnsigned {
							value.LLVMValueRef = llvm.BuildZExt(c.builder, fromValue, typeTo.valueType, "")
						} else if !fromIsUnsigned && !toIsUnsigned {
							value.LLVMValueRef = llvm.BuildSExt(c.builder, fromValue, typeTo.valueType, "")
						} else {
							// Signed to Unsigned or vice versa doesn't require an
							// explicit instruction since the underlying bit format is the same.
							value.LLVMValueRef = fromValue
						}
					} else if fromIsFloat && toIsFloat {
						if fromSize > toSize {
							value.LLVMValueRef = llvm.BuildFPTrunc(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value.LLVMValueRef = llvm.BuildFPExt(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsFloat && toIsInteger {
						if toIsUnsigned {
							value.LLVMValueRef = llvm.BuildFPToSI(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value.LLVMValueRef = llvm.BuildFPToUI(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsInteger && toIsFloat {
						if toIsUnsigned {
							value.LLVMValueRef = llvm.BuildUIToFP(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value.LLVMValueRef = llvm.BuildSIToFP(c.builder, fromValue, typeTo.valueType, "")
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
				value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
			}
		case *types.Pointer:
			otherType := expr.Type().(*types.Basic)
			if otherType.Kind() == types.UnsafePointer {
				value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
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
					value.LLVMValueRef = llvm.GetParam(fn.value, uint(i))
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
			return invalidValue, err
		}

		structType := llvm.TypeOf(structValue)

		// Get the address of the field within the return struct (tuple)
		fieldType, addr := c.structFieldAddress(structValue, structType, expr.Index)

		// Load the value at the address
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.Field:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}

		structType := llvm.TypeOf(structValue)

		// Get the address of the field within the struct
		fieldType, addr := c.structFieldAddress(structValue, structType, expr.Field)

		// Load the value at the address
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.FieldAddr:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}

		var structType llvm.LLVMTypeRef

		if pointerType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
			structType = c.createType(ctx, pointerType.Elem()).valueType
		} else {
			structType = llvm.TypeOf(structValue)
		}

		//Return the address
		_, value.LLVMValueRef = c.structFieldAddress(structValue, structType, expr.Field)
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
			value.LLVMValueRef = llvm.AddGlobal(c.module, globalType.valueType, info.LinkName)
			llvm.SetLinkage(value.LLVMValueRef, llvm.LLVMLinkage(llvm.ExternalLinkage))
		} else {
			value.LLVMValueRef = c.createGlobalValue(ctx, llvm.ConstNull(globalType.valueType), types.Id(c.currentPackage(ctx).Pkg, expr.Name()))
			if !isExported {
				llvm.SetLinkage(value.LLVMValueRef, llvm.LLVMLinkage(llvm.PrivateLinkage))
			}
		}
	case *ssa.Index:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return invalidValue, err
		}

		// Get the address of the element at the index within the array
		var addr llvm.LLVMTypeRef
		var loadElementType llvm.LLVMTypeRef

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())
			loadElementType = elementType.valueType

			// Create a runtime call to retrieve the address of the element at index I
			addr, err = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(arrayValue),
				indexValue,
				c.createTypeDescriptor(ctx, elementType),
			})
			if err != nil {
				return invalidValue, err
			}
		case *types.Basic: // Strings
			loadElementType = llvm.Int8TypeInContext(c.currentContext(ctx))

			// Create a runtime call to retrieve the address of the element at index I
			addr, err = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(arrayValue),
				indexValue,
			})
			if err != nil {
				return invalidValue, err
			}
		default:
			var arrayType llvm.LLVMTypeRef
			loadElementType = llvm.GetElementType(llvm.TypeOf(arrayValue))
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				arrayType = c.createType(ctx, ptrType.Elem()).valueType
			} else {
				arrayType = llvm.TypeOf(arrayValue)
			}

			// Get the address of the element at the index within the array
			addr = llvm.BuildGEP2(c.builder, llvm.GetElementType(arrayType), arrayValue.UnderlyingValue(),
				[]llvm.LLVMValueRef{indexValue}, "")
		}

		// Load the value at the address
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, loadElementType, addr, "")
	case *ssa.IndexAddr:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return invalidValue, err
		}

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())

			// Create a runtime call to retrieve the address of the element at index I
			value.LLVMValueRef, err = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(arrayValue),
				indexValue,
				c.createTypeDescriptor(ctx, elementType),
			})
			if err != nil {
				return invalidValue, err
			}
		case *types.Basic: // Strings
			// Create a runtime call to retrieve the address of the element at index I
			value.LLVMValueRef, err = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(arrayValue),
				indexValue,
			})
			if err != nil {
				return invalidValue, err
			}
		default:
			var arrayType llvm.LLVMTypeRef
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				arrayType = c.createType(ctx, ptrType.Elem()).valueType
			} else {
				arrayType = llvm.TypeOf(arrayValue)
			}

			// Get the address of the element at the index within the array
			value.LLVMValueRef = llvm.BuildGEP2(c.builder, llvm.GetElementType(arrayType), arrayValue.UnderlyingValue(),
				[]llvm.LLVMValueRef{indexValue}, "")
		}
	case *ssa.Lookup:
		panic("not implemented")
	case *ssa.MakeChan:
		panic("not implemented")
	case *ssa.MakeClosure:
		panic("not implemented")
	case *ssa.MakeInterface:
		value.LLVMValueRef, err = c.makeInterface(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}
	case *ssa.MakeSlice:
		lenValue, err := c.createExpression(ctx, expr.Len)
		if err != nil {
			return invalidValue, err
		}

		capValue, err := c.createExpression(ctx, expr.Cap)
		if err != nil {
			return invalidValue, err
		}

		elementType := c.createType(ctx, expr.Type().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)

		// Create the runtime call
		value.LLVMValueRef, err = c.createRuntimeCall(ctx, "makeSlice", []llvm.LLVMValueRef{
			elementTypeDesc,
			lenValue,
			capValue,
		})

		// Load the slice value
		sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, sliceType, value, "")
	case *ssa.MultiConvert:
		panic("not implemented")
	case *ssa.Next:
		panic("not implemented")
	case *ssa.Phi:
		phiType := c.createType(ctx, expr.Type())

		// Build the Phi node operator
		value.LLVMValueRef = llvm.BuildPhi(c.builder, phiType.valueType, "")

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
			return invalidValue, err
		}

		// Get the blocks each edge value belongs to
		// NOTE: Edges[i] is value for Block().Preds[i]
		var blocks []llvm.LLVMBasicBlockRef
		for i, _ := range expr.Edges {
			block := c.blocks[expr.Block().Preds[i]]
			blocks = append(blocks, block)
		}

		// Add the edges
		llvm.AddIncoming(value.LLVMValueRef, edgeValues.Ref(), blocks)
	case *ssa.Slice:
		var low, high, max llvm.LLVMValueRef
		if expr.Low != nil {
			low, err = c.createExpression(ctx, expr.Low)
			if err != nil {
				return invalidValue, err
			}
		}

		if expr.High != nil {
			high, err = c.createExpression(ctx, expr.High)
			if err != nil {
				return invalidValue, err
			}
		}

		if expr.Max != nil {
			max, err = c.createExpression(ctx, expr.Max)
			if err != nil {
				return invalidValue, err
			}
		}

		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
		}

		array := arrayValue.UnderlyingValue()

		var elementType llvm.LLVMTypeRef
		numElements := uint64(0)

		switch t := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			elementType = c.createType(ctx, t.Elem()).valueType
		case *types.Basic:
			elementType = llvm.Int8TypeInContext(c.currentContext(ctx))
		case *types.Pointer:
			if tt, ok := t.Elem().Underlying().(*types.Array); ok {
				elementType = c.createType(ctx, tt.Elem()).valueType
				numElements = uint64(tt.Len())
			} else {
				panic("invalid pointer type")
			}
		}

		value.LLVMValueRef = c.createSlice(ctx, array, elementType, numElements, low, high, max)
	case *ssa.SliceToArrayPointer:
		panic("not implemented")
	case *ssa.TypeAssert:
		x, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return invalidValue, err
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
			return invalidValue, err
		}

		// Get addresses of return value
		result = c.addressOf(result)
		objAddr := llvm.BuildStructGEP2(c.builder, c.ptrType.valueType, result, 0, "")
		okAddr := llvm.BuildStructGEP2(c.builder, llvm.Int1TypeInContext(c.currentContext(ctx)), result, 1, "")

		// TODO: There definitely more involved than this. Gonna try to
		//       implement the semantics in Go code rather than hardcode it
		//       here. I should be able to load the resulting object from
		//       the address obtained from the runtime call.
		objValue := llvm.BuildLoad2(c.builder, newType.valueType, objAddr, "")

		if expr.CommaOk {
			// Return the obj and the status
			value.LLVMValueRef = llvm.ConstStruct([]llvm.LLVMValueRef{
				llvm.GetUndef(newType.valueType),
				llvm.GetUndef(llvm.Int1Type()),
			}, false)

			okValue := llvm.BuildLoad2(c.builder, newType.valueType, okAddr, "")

			value.LLVMValueRef = llvm.BuildInsertValue(c.builder, value.LLVMValueRef, objValue, uint(0), "")
			value.LLVMValueRef = llvm.BuildInsertValue(c.builder, value.LLVMValueRef, okValue, uint(1), "")
		} else {
			value.LLVMValueRef = objValue
		}
	case *ssa.UnOp:
		value.LLVMValueRef, err = c.createUpOp(ctx, expr)
	}

	if value.LLVMValueRef == nil {
		panic("nil value")
	}

	// Cache the value
	c.values[expr] = value

	return value, nil
}
