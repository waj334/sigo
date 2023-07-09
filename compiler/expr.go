package compiler

import (
	"context"
	"go/constant"
	"go/types"
	"math"

	"golang.org/x/tools/go/ssa"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createExpression(ctx context.Context, expr ssa.Value) (value Value) {
	c.printf(Debug, "Processing expression %T: %s\n", expr, expr.String())

	// Check if this value was cached
	if value, ok := c.values[expr]; ok {
		// Return the cached value
		c.println(Debug, "returning cached value")
		return value
	}

	// initialize the value
	value.cc = c
	value.spec = expr

	// Evaluate the expression
	switch expr := expr.(type) {
	case *ssa.Alloc:
		// Get the type of which memory will be allocated for
		elementType := c.createType(ctx, expr.Type().Underlying().(*types.Pointer).Elem())

		// NOTE: Some stack allocations will be moved to the heap later if they
		//       are too big for the stack.
		// Get the size of the type
		size := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType.valueType)

		currentBlock := llvm.GetInsertBlock(c.builder)

		if expr.Heap {
			// Create the alloca to hold the address on the stack
			value.ref = c.createAlloca(ctx, c.ptrType.valueType, expr.Comment)

			// Mark this value as one that is on the heap
			value.heap = true

			// Next, create the debug info. This heap allocation will be treated differently
			c.createVariable(ctx, expr.Comment, value, expr.Type().Underlying())

			// Create the runtime call to allocate some memory on the heap
			next := llvm.GetNextInstruction(value.ref)
			if next == nil {
				llvm.PositionBuilderAtEnd(c.builder, llvm.GetInstructionParent(value.ref))
			} else {
				llvm.PositionBuilderBefore(c.builder, next)
			}
			obj := c.createRuntimeCall(ctx, "alloc",
				[]llvm.LLVMValueRef{llvm.ConstInt(c.uintptrType.valueType, size, false)})

			// Store the address at the alloc
			llvm.BuildStore(c.builder, obj, value.ref)
			llvm.PositionBuilderAtEnd(c.builder, currentBlock)
		} else {
			// Create an alloca to hold the value on the stack
			value.ref = c.createAlloca(ctx, elementType.valueType, expr.Comment)
			c.createVariable(ctx, expr.Comment, value, elementType.spec)
			value.heap = false

			// Zero-initialize the stack variable
			next := llvm.GetNextInstruction(value.ref)
			if next == nil {
				llvm.PositionBuilderAtEnd(c.builder, llvm.GetInstructionParent(value.ref))
			} else {
				llvm.PositionBuilderBefore(c.builder, next)
			}
			llvm.BuildStore(c.builder, llvm.ConstNull(elementType.valueType), value.ref)
			llvm.PositionBuilderAtEnd(c.builder, currentBlock)
		}
	case *ssa.BinOp:
		value.ref = c.createBinOp(ctx, expr)
	case *ssa.Call:
		if c.isIntrinsicCall(expr) {
			// Replace calls to intrinsic functions
			value = c.createIntrinsic(ctx, expr)
		} else if expr.Call.IsInvoke() {
			// Get the interface value
			interfaceValue := c.createExpression(ctx, expr.Call.Value).UnderlyingValue(ctx)

			// Look up the function to be called
			fnPtr := c.createRuntimeCall(ctx, "interfaceLookUp", []llvm.LLVMValueRef{
				interfaceValue,
				llvm.ConstInt(llvm.Int32TypeInContext(
					c.currentContext(ctx)), uint64(c.computeFunctionHash(expr.Call.Method)), false),
			})

			// Load the value pointer from the interface
			valuePtr := llvm.BuildExtractValue(c.builder, interfaceValue, 1, "")
			args := c.createValues(ctx, expr.Call.Args).Ref(ctx)
			args = append([]llvm.LLVMValueRef{valuePtr}, args...)

			// Call the concrete method
			fnType := c.createFunctionType(ctx, expr.Call.Signature(), false)
			value.ref = llvm.BuildCall2(c.builder, fnType, fnPtr, args, "")
		} else if builtin, ok := expr.Call.Value.(*ssa.Builtin); ok {
			value.ref = c.createBuiltinCall(ctx, builtin, expr.Call.Args)
		} else {
			var fnType llvm.LLVMTypeRef
			var fn llvm.LLVMValueRef
			var args []llvm.LLVMValueRef

			switch callee := expr.Call.Value.(type) {
			case *ssa.Function:
				fn = c.createExpression(ctx, expr.Call.Value).UnderlyingValue(ctx)
				args = c.createValues(ctx, expr.Call.Args).Ref(ctx)
				fnType = c.createFunctionType(ctx, callee.Signature, len(callee.FreeVars) > 0)
			case *ssa.MakeClosure:
				panic("not implemented")
			default:
				fn = c.createExpression(ctx, callee).UnderlyingValue(ctx)
				fnType = c.createFunctionType(ctx, callee.Type().Underlying().(*types.Signature), false)
				args = c.createValues(ctx, expr.Call.Args).Ref(ctx)
			}

			// Check the arguments
			paramTypes := llvm.GetParamTypes(fnType)
			for i, paramType := range paramTypes {
				argType := llvm.TypeOf(args[i])
				if !llvm.TypeIsEqual(argType, paramType) {
					panic(llvm.PrintTypeToString(argType) + " != " + llvm.PrintTypeToString(paramType))
				}
			}

			// Create the call
			value.ref = llvm.BuildCall2(c.builder, fnType, fn, args, "")
		}
	case *ssa.Const:
		if expr.Value == nil {
			constType := c.createType(ctx, expr.Type())
			value.ref = llvm.ConstNull(constType.valueType)
		} else {
			if basicType, ok := expr.Type().Underlying().(*types.Basic); ok && basicType.Kind() == types.UntypedInt {
				bitLen := int(math.Ceil(float64(constant.BitLen(expr.Value)) / 8))
				if intVal, ok := constant.Int64Val(expr.Value); ok {
					switch bitLen {
					case 8:
						value.ref = llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 4:
						value.ref = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 2:
						value.ref = llvm.ConstInt(llvm.Int16TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 1:
						value.ref = llvm.ConstInt(llvm.Int8TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					default:
						panic("incompatible untyped int size")
					}
				} else if uintVal, ok := constant.Uint64Val(expr.Value); ok {
					switch bitLen {
					case 8:
						value.ref = llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 4:
						value.ref = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 2:
						value.ref = llvm.ConstInt(llvm.Int16TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 1:
						value.ref = llvm.ConstInt(llvm.Int8TypeInContext(c.currentContext(ctx)), uintVal, false)
					default:
						panic("incompatible untyped int size")
					}
				}
			} else {
				constType := c.createType(ctx, expr.Type())
				switch t := expr.Type().Underlying().(type) {
				case *types.Basic:
					switch {
					case t.Kind() == types.UnsafePointer:
						value.ref = llvm.ConstInt(c.uintptrType.valueType, expr.Uint64(), false)
						value.ref = llvm.BuildIntToPtr(c.builder, value.ref, constType.valueType, "")
					case t.Info()&types.IsInteger != 0:
						if t.Info()&types.IsUnsigned != 0 {
							value.ref = llvm.ConstInt(constType.valueType, expr.Uint64(), false)
						} else {
							value.ref = llvm.ConstInt(constType.valueType, uint64(expr.Int64()), false)
						}
					case t.Info()&types.IsString != 0:
						strValue := constant.StringVal(expr.Value)
						value.ref = c.createConstantString(ctx, strValue)
					case t.Info()&types.IsBoolean != 0:
						value.ref = c.createConstBool(ctx, constant.BoolVal(expr.Value))
					case t.Info()&types.IsFloat != 0:
						value.ref = llvm.ConstReal(constType.valueType, expr.Float64())
					default:
						panic("not implemented")
					}
				}
			}
		}
	case *ssa.Convert:
		typeTo := c.createType(ctx, expr.Type().Underlying())
		fromValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)

		switch typeX := expr.X.Type().Underlying().(type) {
		case *types.Basic:
			switch otherType := expr.Type().Underlying().(type) {
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
						value.ref = llvm.BuildPtrToInt(c.builder, fromValue, typeTo.valueType, "")
					} else {
						value.ref = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
					}
				case types.Uintptr:
					if otherType.Kind() == types.UnsafePointer {
						value.ref = llvm.BuildIntToPtr(c.builder, fromValue, typeTo.valueType, "")
						break
					}
					fallthrough
				default:
					if fromIsInteger && toIsInteger {
						value.ref = llvm.BuildIntCast2(c.builder, fromValue, typeTo.valueType, fromIsUnsigned, "")
					} else if fromIsFloat && toIsFloat {
						value.ref = llvm.BuildFPCast(c.builder, fromValue, typeTo.valueType, "")
					} else if fromIsFloat && toIsInteger {
						if toIsUnsigned {
							value.ref = llvm.BuildFPToSI(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value.ref = llvm.BuildFPToUI(c.builder, fromValue, typeTo.valueType, "")
						}
					} else if fromIsInteger && toIsFloat {
						if fromIsUnsigned {
							value.ref = llvm.BuildUIToFP(c.builder, fromValue, typeTo.valueType, "")
						} else {
							value.ref = llvm.BuildSIToFP(c.builder, fromValue, typeTo.valueType, "")
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
				value.ref = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
			case *types.Slice:
				value.ref = c.createRuntimeCall(ctx, "stringToSlice", []llvm.LLVMValueRef{fromValue})
			}
		case *types.Pointer:
			otherType := expr.Type().(*types.Basic)
			if otherType.Kind() == types.UnsafePointer {
				value.ref = llvm.BuildPointerCast(c.builder, fromValue, typeTo.valueType, "")
			} else {
				panic("not implemented")
			}
		case *types.Slice:
			panic("not implemented")
		}
	case *ssa.Parameter:
		if fn, ok := c.functions[expr.Parent()]; ok {
			// Locate the parameter in the function
			for i, param := range expr.Parent().Params {
				if param == expr {
					value.ref = llvm.GetParam(fn.value, uint(i))
				}
			}
		} else {
			panic("function does not exist")
		}
	case *ssa.ChangeInterface:
		interfaceType := c.createRuntimeType(ctx, "_interface").valueType
		concreteType := c.createTypeDescriptor(ctx, c.createType(ctx, expr.Type()))
		interfaceValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		concreteValue := llvm.BuildExtractValue(c.builder, interfaceValue, 1, "")
		value.ref = llvm.GetUndef(interfaceType)
		value.ref = llvm.BuildInsertValue(c.builder, value.ref, concreteType, 0, "")
		value.ref = llvm.BuildInsertValue(c.builder, value.ref, concreteValue, 1, "")
	case *ssa.ChangeType:
		// TODO: Actually cast the types
		value.ref = c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
	case *ssa.Extract:
		// Get the return struct (tuple)
		structValue := c.createExpression(ctx, expr.Tuple).UnderlyingValue(ctx)
		value.ref = llvm.BuildExtractValue(c.builder, structValue, uint(expr.Index), "")
	case *ssa.Field:
		structValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		value.ref = llvm.BuildExtractValue(c.builder, structValue, uint(expr.Field), "")
	case *ssa.FieldAddr:
		var structType llvm.LLVMTypeRef
		structValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)

		pointerType := expr.X.Type().Underlying().(*types.Pointer)

		// Nil check the pointer
		c.createRuntimeCall(ctx, "nilCheck", []llvm.LLVMValueRef{structValue})
		structType = c.createType(ctx, pointerType.Elem()).valueType

		// Create a GEP to get the address of the field within the struct
		value.ref = llvm.BuildStructGEP2(
			c.builder,
			structType,
			structValue,
			uint(expr.Field), "struct_field_addr")
	case *ssa.FreeVar:
		fnObj := c.functions[expr.Parent()]
		varType := c.createType(ctx, expr.Type())

		// The last parameter of the function is the struct containing free var values
		state := llvm.GetParam(fnObj.value, llvm.CountParamTypes(fnObj.llvmType)-1)

		// FreeVars are stored in the state struct in the order that they appear in ssa.Function
		for i, freeVar := range fnObj.def.FreeVars {
			if freeVar == expr {
				// Get the address of the value to be loaded from the struct
				addr := llvm.BuildStructGEP2(c.builder, fnObj.stateType, state, uint(i), "free_var_addr")

				// Load the value from the struct
				value.ref = llvm.BuildLoad2(c.builder, varType.valueType, addr, "free_var")

				// Stop
				break
			}
		}
	case *ssa.Function:
		// Check the function cache first
		fn, ok := c.functions[expr]
		if !ok {
			fn = c.createFunction(ctx, expr)
		}

		// Return a pointer to the function
		value.ref = fn.value
	case *ssa.Global:
		// Create a global value
		globalType := c.createType(ctx, expr.Object().Type())
		value.linkname = c.symbolName(expr.Pkg.Pkg, expr.Object().Name())

		info := c.options.GetSymbolInfo(value.linkname)
		value.extern = info.ExternalLinkage
		value.exported = info.Exported || expr.Object().Exported()
		if len(info.LinkName) > 0 {
			value.linkname = info.LinkName
		}

		// Cannot be both exported and external
		if value.exported && value.extern {
			panic("global cannot be both external and exported")
		}

		if value.extern {
			// Look for the global value within the module
			value.ref = llvm.GetNamedGlobal(c.module, value.linkname)
			if value.ref == nil {
				// Look for a function with this name
				value.ref = llvm.GetNamedFunction(c.module, value.linkname)
				if value.ref != nil {
					value.ref = llvm.BuildBitCast(c.builder, value.ref, globalType.valueType, "")
					value.ref = llvm.BuildPointerCast(c.builder, value.ref, globalType.valueType, "")
				} else {
					// Create a global with external linkage to some variable with the specified link name.
					value.ref = llvm.AddGlobal(c.module, globalType.valueType, value.linkname)
					llvm.SetUnnamedAddr(value.ref, true)
					llvm.SetLinkage(value.ref, llvm.ExternalLinkage)
				}
			}
		} else if expr.Pkg != c.currentPackage(ctx) {
			// Create a extern global value
			value.ref = llvm.AddGlobal(c.module, globalType.valueType, value.linkname)
			llvm.SetUnnamedAddr(value.ref, true)
			llvm.SetLinkage(value.ref, llvm.ExternalLinkage)
		} else {
			value.ref = c.createGlobalValue(ctx, llvm.ConstNull(globalType.valueType), value.linkname)
			llvm.SetLinkage(value.ref, llvm.CommonLinkage)
		}
	case *ssa.Index:
		arrayValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		indexValue := c.createExpression(ctx, expr.Index).UnderlyingValue(ctx)

		// Get the address of the element at the index within the array
		var addr llvm.LLVMValueRef
		var loadElementType llvm.LLVMTypeRef

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())
			loadElementType = elementType.valueType

			// Create a runtime call to retrieve the address of the element at index I
			addr = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				arrayValue,
				indexValue,
				c.createTypeDescriptor(ctx, elementType),
			})
		case *types.Basic: // Strings
			loadElementType = llvm.Int8TypeInContext(c.currentContext(ctx))

			// Create a runtime call to retrieve the address of the element at index I
			addr = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				arrayValue,
				indexValue,
			})
		default:
			var arrayType llvm.LLVMTypeRef
			loadElementType = llvm.GetElementType(llvm.TypeOf(arrayValue))
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				// nil pointer check the value
				c.createRuntimeCall(ctx, "nilCheck", []llvm.LLVMValueRef{arrayValue})
				arrayType = c.createType(ctx, ptrType.Elem()).valueType
			} else {
				arrayType = llvm.TypeOf(arrayValue)
			}

			// Get the address of the element at the index within the array
			addr = llvm.BuildGEP2(c.builder, llvm.GetElementType(arrayType), arrayValue,
				[]llvm.LLVMValueRef{indexValue}, "")
		}

		// Load the value at the address
		value.ref = llvm.BuildLoad2(c.builder, loadElementType, addr, "")
	case *ssa.IndexAddr:
		arrayValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		indexValue := c.createExpression(ctx, expr.Index).UnderlyingValue(ctx)
		indexType := expr.Index.Type().Underlying().(*types.Basic)

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())

			// Create a runtime call to retrieve the address of the element at index I
			value.ref = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				arrayValue,
				llvm.BuildIntCast2(c.builder,
					indexValue,
					llvm.Int32TypeInContext(c.currentContext(ctx)),
					indexType.Info()&types.IsUnsigned == 0,
					""),
				c.createTypeDescriptor(ctx, elementType),
			})
		case *types.Basic: // Strings
			// Create a runtime call to retrieve the address of the element at index I
			value.ref = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				arrayValue,
				indexValue,
			})
		default:
			var arrayType llvm.LLVMTypeRef
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				arrayType = c.createType(ctx, ptrType.Elem()).valueType
			} else {
				arrayType = llvm.TypeOf(arrayValue)
			}

			// Get the address of the element at the index within the array
			value.ref = llvm.BuildGEP2(c.builder, llvm.GetElementType(arrayType), arrayValue,
				[]llvm.LLVMValueRef{indexValue}, "")
		}
	case *ssa.Lookup:
		mapValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		keyValue := c.createExpression(ctx, expr.Index).UnderlyingValue(ctx)
		elemType := c.createType(ctx, expr.Type()).valueType

		// Create space on the stack to store the value returned by the lookup
		result := c.createAlloca(ctx, elemType, "")

		// Create the runtime call to perform the map lookup
		okValue := c.createRuntimeCall(ctx, "mapLookup", []llvm.LLVMValueRef{
			mapValue,
			c.addressOf(ctx, keyValue),
			result,
		})

		// Load the result
		result = llvm.BuildLoad2(c.builder, elemType, result, "")

		if expr.CommaOk {
			// Create an aggregate to return
			resultType := llvm.StructTypeInContext(
				c.currentContext(ctx),
				[]llvm.LLVMTypeRef{
					elemType,
					c.int1Type(ctx),
				}, false)
			value.ref = llvm.GetUndef(resultType)
			value.ref = llvm.BuildInsertValue(c.builder, value.ref, result, 0, "")
			value.ref = llvm.BuildInsertValue(c.builder, value.ref, okValue, 1, "")
		} else {
			value.ref = result
		}
	case *ssa.MakeChan:
		chanType := expr.Type().(*types.Chan)
		value.ref = c.createRuntimeCall(ctx, "channelMake", []llvm.LLVMValueRef{
			c.createTypeDescriptor(ctx, c.createType(ctx, chanType)),
			c.createExpression(ctx, expr.Size).UnderlyingValue(ctx),
		})
	case *ssa.MakeClosure:
		fnObj := c.functions[expr.Fn.(*ssa.Function)]

		// Create the state struct
		value.ref = c.createAlloca(ctx, fnObj.stateType, "")
		for i, bound := range expr.Bindings {
			val := c.createExpression(ctx, bound).UnderlyingValue(ctx)
			addr := llvm.BuildStructGEP2(c.builder, fnObj.stateType, value.ref, uint(i), "")
			llvm.BuildStore(c.builder, val, addr)
		}
	case *ssa.MakeInterface:
		value.ref = c.makeInterface(ctx, expr.X)
	case *ssa.MakeMap:
		mapType := expr.Type().Underlying().(*types.Map)
		keyType := c.createTypeDescriptor(ctx, c.createType(ctx, mapType.Key()))
		valueType := c.createTypeDescriptor(ctx, c.createType(ctx, mapType.Elem()))
		var capacityValue llvm.LLVMValueRef
		if expr.Reserve != nil {
			capacityValue = c.createExpression(ctx, expr.Reserve).UnderlyingValue(ctx)
		} else {
			capacityValue = llvm.ConstInt(c.int32Type(ctx), 0, false)
		}

		// TODO: Handle reserve value
		value.ref = c.createRuntimeCall(ctx, "mapMake", []llvm.LLVMValueRef{keyType, valueType, capacityValue})
	case *ssa.MakeSlice:
		lenValue := c.createExpression(ctx, expr.Len).UnderlyingValue(ctx)
		lenValueType := expr.Len.Type().Underlying().(*types.Basic)
		capValue := c.createExpression(ctx, expr.Cap).UnderlyingValue(ctx)
		capValueType := expr.Cap.Type().Underlying().(*types.Basic)
		elementType := c.createType(ctx, expr.Type().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)

		// Create the runtime call
		value.ref = c.createRuntimeCall(ctx, "sliceMake", []llvm.LLVMValueRef{
			elementTypeDesc,
			llvm.BuildIntCast2(c.builder,
				lenValue,
				llvm.Int32TypeInContext(c.currentContext(ctx)),
				lenValueType.Info()&types.IsUnsigned == 0,
				""),
			llvm.BuildIntCast2(c.builder,
				capValue,
				llvm.Int32TypeInContext(c.currentContext(ctx)),
				capValueType.Info()&types.IsUnsigned == 0,
				""),
		})
	case *ssa.MultiConvert:
		//lhsValue := c.createExpression(ctx, expr.X)
		panic("not implemented")
	case *ssa.Next:
		iter := c.createExpression(ctx, expr.Iter).UnderlyingValue(ctx)

		// Create the respective runtime call
		if expr.IsString {
			value.ref = c.createRuntimeCall(ctx, "stringRange", []llvm.LLVMValueRef{iter})
		} else {
			panic("map type not implemented")
		}
	case *ssa.Phi:
		phiType := c.createType(ctx, expr.Type())

		// Build the Phi node operator
		value.ref = llvm.BuildPhi(c.builder, phiType.valueType, "")

		// Cache the PHI value now to prevent a stack overflow in the call to createValues below
		if _, ok := c.values[expr]; ok {
			panic("PHI node value already generated")
		}
		c.values[expr] = value

		// Create values for each of the Phi node's edges
		edgeValues := c.createValues(ctx, expr.Edges)

		// Get the blocks each edge value belongs to
		// NOTE: Edges[i] is value for Block().Preds[i]
		var blocks []llvm.LLVMBasicBlockRef
		for i, _ := range expr.Edges {
			block := c.blocks[expr.Block().Preds[i]]
			blocks = append(blocks, block)
		}

		// Add the edges
		llvm.AddIncoming(value.ref, edgeValues.Ref(ctx), blocks)
	case *ssa.Range:
		containerVal := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		switch expr.X.Type().Underlying().(type) {
		case *types.Basic:
			// Create a struct representing the string/map iterator
			strType := llvm.StructTypeInContext(c.currentContext(ctx), []llvm.LLVMTypeRef{
				c.createType(ctx, expr.X.Type().Underlying()).valueType,
				c.int32Type(ctx),
			}, false)
			value.ref = c.createAlloca(ctx, strType, "string_iterator")

			// Set the range parameter values in the struct
			llvm.BuildStore(c.builder, containerVal, llvm.BuildStructGEP2(c.builder, strType, value.ref, 0, "str_addr"))
			llvm.BuildStore(c.builder, llvm.ConstInt(c.int64Type(ctx), 0, false), llvm.BuildStructGEP2(c.builder, strType, value.ref, 1, "index_arr"))
		case *types.Map:
			panic("map type not implemented")
		}
	case *ssa.Select:
		chanValues := make([]llvm.LLVMValueRef, len(expr.States))
		sendValues := make([]llvm.LLVMValueRef, len(expr.States))
		directionValues := make([]llvm.LLVMValueRef, len(expr.States))
		results := make([]llvm.LLVMValueRef, len(expr.States))

		// Create each of the channel values
		for i := range expr.States {
			chanType := expr.States[i].Chan.Type().(*types.Chan)
			chanValues[i] = c.createExpression(ctx, expr.States[i].Chan).UnderlyingValue(ctx)
			directionValues[i] = c.createConstBool(ctx, expr.States[i].Dir == types.SendOnly)
			results[i] = c.createAlloca(ctx, c.createType(ctx, chanType.Elem()).valueType, "")
			if expr.States[i].Send != nil {
				sendValues[i] = c.createExpression(ctx, expr.States[i].Send).UnderlyingValue(ctx)
			} else {
				sendValues[i] = llvm.ConstNull(c.ptrType.valueType)
			}
		}

		readyCases := llvm.GetUndef(c.createRuntimeType(ctx, "_slice").valueType)
		readyCases = llvm.BuildInsertValue(c.builder, readyCases, c.createArrayAlloca(ctx, c.int32Type(ctx), uint64(len(expr.States)), "ready_cases_arr"), 0, "")
		readyCases = llvm.BuildInsertValue(c.builder, readyCases, llvm.ConstInt(c.int32Type(ctx), uint64(len(expr.States)), false), 1, "")
		readyCases = llvm.BuildInsertValue(c.builder, readyCases, llvm.ConstInt(c.int32Type(ctx), uint64(len(expr.States)), false), 2, "")

		// Create the runtime call
		selectResult := c.createRuntimeCall(ctx, "channelSelect", []llvm.LLVMValueRef{
			c.createSliceFromValues(ctx, chanValues),
			c.createSliceFromValues(ctx, results),
			c.createSliceFromValues(ctx, directionValues),
			c.createSliceFromValues(ctx, sendValues),
			c.addressOf(ctx, readyCases),
			c.createConstBool(ctx, !expr.Blocking),
		})

		indexValue := llvm.BuildExtractValue(c.builder, selectResult, 0, "")
		okValue := llvm.BuildExtractValue(c.builder, selectResult, 1, "")

		// Create the result tuple
		value.ref = llvm.GetUndef(c.createType(ctx, expr.Type()).valueType)
		value.ref = llvm.BuildInsertValue(c.builder, value.ref, indexValue, 0, "")
		value.ref = llvm.BuildInsertValue(c.builder, value.ref, okValue, 1, "")
		for i, r := range results {
			v := llvm.BuildLoad2(c.builder, llvm.GetAllocatedType(r), r, "")
			value.ref = llvm.BuildInsertValue(c.builder, value.ref, v, uint(2+i), "")
		}
	case *ssa.Slice:
		var low, high, max llvm.LLVMValueRef
		neg := llvm.ConstIntOfString(llvm.Int32TypeInContext(c.currentContext(ctx)), "-1", 10)
		if expr.Low != nil {
			low = c.createExpression(ctx, expr.Low).UnderlyingValue(ctx)
			low = llvm.BuildIntCast2(c.builder, low, c.int32Type(ctx), true, "")
		} else {
			low = neg
		}

		if expr.High != nil {
			high = c.createExpression(ctx, expr.High).UnderlyingValue(ctx)
			high = llvm.BuildIntCast2(c.builder, high, c.int32Type(ctx), true, "")
		} else {
			high = neg
		}

		if expr.Max != nil {
			max = c.createExpression(ctx, expr.Max).UnderlyingValue(ctx)
			max = llvm.BuildIntCast2(c.builder, max, c.int32Type(ctx), true, "")
		} else {
			max = neg
		}

		array := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)

		switch t := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			info := c.createTypeDescriptor(ctx, c.createType(ctx, expr.X.Type()))
			value.ref = c.createRuntimeCall(ctx, "sliceReslice", []llvm.LLVMValueRef{
				array, info, low, high, max,
			})
		case *types.Basic:
			value.ref = c.createRuntimeCall(ctx, "stringSlice", []llvm.LLVMValueRef{array, low, high})
		case *types.Pointer:
			if tt, ok := t.Elem().Underlying().(*types.Array); ok {
				info := c.createTypeDescriptor(ctx, c.createType(ctx, tt))
				value.ref = c.createRuntimeCall(ctx, "sliceAddr", []llvm.LLVMValueRef{array, info, low, high})
			} else {
				panic("invalid pointer type")
			}
		}
	case *ssa.SliceToArrayPointer:
		sliceValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		value.ref = llvm.BuildExtractValue(c.builder, sliceValue, 0, "slice_arr")
	case *ssa.TypeAssert:
		x := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		newType := c.createType(ctx, expr.AssertedType)
		commaOk := 0
		if expr.CommaOk {
			commaOk = 1
		}

		// The value needs to be passed by address. Create an alloca for it and store the value
		xAlloca := c.addressOf(ctx, x)

		// create the runtime call
		result := c.createRuntimeCall(ctx, "interfaceAssert", []llvm.LLVMValueRef{
			xAlloca,
			c.createTypeDescriptor(ctx, c.createType(ctx, expr.X.Type())),
			c.createTypeDescriptor(ctx, newType),
			llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), uint64(commaOk), false),
		})

		ptr := llvm.BuildExtractValue(c.builder, result, 0, "")
		ok := llvm.BuildExtractValue(c.builder, result, 1, "")

		// Load the value from the pointer
		assertedValue := llvm.BuildLoad2(c.builder, newType.valueType, ptr, "")

		if expr.CommaOk {
			// Return the obj and the status
			valueType := llvm.StructType([]llvm.LLVMTypeRef{llvm.TypeOf(assertedValue), llvm.TypeOf(ok)}, false)
			value.ref = llvm.GetUndef(valueType)
			value.ref = llvm.BuildInsertValue(c.builder, value.ref, assertedValue, 0, "")
			value.ref = llvm.BuildInsertValue(c.builder, value.ref, ok, 1, "")
		} else {
			value.ref = assertedValue
		}
	case *ssa.UnOp:
		value.ref = c.createUnOp(ctx, expr)
	default:
		panic("unimplemented expression type: " + expr.String())
	}

	if value.ref == nil {
		panic("nil value")
	}

	// Cache the value
	c.values[expr] = value

	return value
}
