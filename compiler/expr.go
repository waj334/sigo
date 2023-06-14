package compiler

import (
	"context"
	"go/constant"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"math"
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

	// Get the current debug location to restore to when this instruction is done
	currentDbgLoc := c.currentDbgLocation(ctx)
	defer llvm.SetCurrentDebugLocation2(c.builder, currentDbgLoc)

	// Change the current debug location to that of the instruction being processed
	scope, file := c.instructionScope(expr)
	var location token.Position
	if file != nil {
		location = file.Position(expr.Pos())
	}
	dbgLoc := llvm.DIBuilderCreateDebugLocation(
		c.currentContext(ctx),
		uint(location.Line),
		uint(location.Column),
		scope,
		nil)

	ctx = context.WithValue(ctx, currentDbgLocationKey{}, dbgLoc)
	llvm.SetCurrentDebugLocation2(c.builder, dbgLoc)

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

		if expr.Heap {
			// Create the alloca to hold the address on the stack
			value.LLVMValueRef = c.createAlloca(ctx, c.ptrType.valueType, expr.Comment)

			// Mark this value as one that is on the heap
			value.heap = true

			// Next, create the debug info. This heap allocation will be treated differently
			value = c.createVariable(ctx, expr.Comment, value, expr.Type().Underlying())

			// Create the runtime call to allocate some memory on the heap
			obj := c.createRuntimeCall(ctx, "alloc",
				[]llvm.LLVMValueRef{llvm.ConstInt(c.uintptrType.valueType, size, false)})

			// Store the address at the alloc
			llvm.BuildStore(c.builder, obj, value)
		} else {
			// Create an alloca to hold the value on the stack
			value.LLVMValueRef = c.createAlloca(ctx, elementType.valueType, expr.Comment)
			value = c.createVariable(ctx, expr.Comment, value, elementType.spec)
			value.heap = false

			// Zero-initialize the stack variable
			llvm.BuildStore(c.builder, llvm.ConstNull(elementType.valueType), value)
		}
	case *ssa.BinOp:
		value.LLVMValueRef = c.createBinOp(ctx, expr)
	case *ssa.Call:
		if c.isIntrinsicCall(expr) {
			// Replace calls to intrinsic functions
			value = c.createIntrinsic(ctx, expr)
		} else if expr.Call.IsInvoke() {
			// Get the interface value
			interfaceValue := c.createExpression(ctx, expr.Call.Value).UnderlyingValue(ctx)

			// Look up the function to be called
			fnPtr := c.createRuntimeCall(ctx, "interfaceLookUp", []llvm.LLVMValueRef{
				c.addressOf(ctx, interfaceValue),
				llvm.ConstInt(llvm.Int32TypeInContext(
					c.currentContext(ctx)), uint64(c.computeFunctionHash(expr.Call.Method)), false),
			})

			// Load the value pointer from the interface
			valuePtr := llvm.BuildExtractValue(c.builder, interfaceValue, 1, "")
			args := c.createValues(ctx, expr.Call.Args).Ref(ctx)
			args = append([]llvm.LLVMValueRef{valuePtr}, args...)

			// Call the concrete method
			fnType := c.createFunctionType(ctx, expr.Call.Signature(), false)
			value.LLVMValueRef = llvm.BuildCall2(c.builder, fnType, fnPtr, args, "")
		} else if builtin, ok := expr.Call.Value.(*ssa.Builtin); ok {
			value.LLVMValueRef = c.createBuiltinCall(ctx, builtin, expr.Call.Args)
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
			value.LLVMValueRef = llvm.BuildCall2(c.builder, fnType, fn, args, "")
		}
	case *ssa.Const:
		if expr.Value == nil {
			constType := c.createType(ctx, expr.Type())
			value.LLVMValueRef = llvm.ConstNull(constType.valueType)
		} else {
			if basicType, ok := expr.Type().Underlying().(*types.Basic); ok && basicType.Kind() == types.UntypedInt {
				bitLen := int(math.Ceil(float64(constant.BitLen(expr.Value)) / 8))
				if intVal, ok := constant.Int64Val(expr.Value); ok {
					switch bitLen {
					case 8:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 4:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 2:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int16TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					case 1:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int8TypeInContext(c.currentContext(ctx)), uint64(intVal), false)
					default:
						panic("incompatible untyped int size")
					}
				} else if uintVal, ok := constant.Uint64Val(expr.Value); ok {
					switch bitLen {
					case 8:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int64TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 4:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 2:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int16TypeInContext(c.currentContext(ctx)), uintVal, false)
					case 1:
						value.LLVMValueRef = llvm.ConstInt(llvm.Int8TypeInContext(c.currentContext(ctx)), uintVal, false)
					default:
						panic("incompatible untyped int size")
					}
				}
			} else {
				constType := c.createType(ctx, expr.Type())
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
					intVal, _ := constant.Int64Val(expr.Value)
					if llvm.GetTypeKind(constType.valueType) == llvm.PointerTypeKind {
						constVal := llvm.ConstInt(c.uintptrType.valueType, uint64(intVal), false)
						value.LLVMValueRef = llvm.BuildIntToPtr(c.builder, constVal, constType.valueType, "")
					} else {
						value.LLVMValueRef = llvm.ConstInt(constType.valueType, uint64(intVal), false)
					}
				case constant.Float:
					constVal, _ := constant.Float64Val(expr.Value)
					value.LLVMValueRef = llvm.ConstReal(constType.valueType, constVal)
				case constant.Complex:
					panic("not implemented")
				default:
					panic("unknown default value")
				}
			}
		}
	case *ssa.Convert:
		typeTo := c.createType(ctx, expr.Type().Underlying())
		fromValue := c.createExpression(ctx, expr.X)

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
						value.LLVMValueRef = llvm.BuildPtrToInt(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
					} else {
						value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
					}
				case types.Uintptr:
					if otherType.Kind() == types.UnsafePointer {
						value.LLVMValueRef = llvm.BuildIntToPtr(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
						break
					}
					fallthrough
				default:
					if fromIsInteger && toIsInteger {
						value.LLVMValueRef = llvm.BuildIntCast2(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, fromIsUnsigned, "")
					} else if fromIsFloat && toIsFloat {
						value.LLVMValueRef = llvm.BuildFPCast(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
					} else if fromIsFloat && toIsInteger {
						if toIsUnsigned {
							value.LLVMValueRef = llvm.BuildFPToSI(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
						} else {
							value.LLVMValueRef = llvm.BuildFPToUI(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
						}
					} else if fromIsInteger && toIsFloat {
						if toIsUnsigned {
							value.LLVMValueRef = llvm.BuildUIToFP(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
						} else {
							value.LLVMValueRef = llvm.BuildSIToFP(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
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
				value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
			case *types.Slice:
				sliceValue := c.createRuntimeCall(ctx, "sliceString", []llvm.LLVMValueRef{fromValue.UnderlyingValue(ctx)})

				// Load the resulting slice
				value.LLVMValueRef = llvm.BuildLoad2(c.builder, typeTo.valueType, c.addressOf(ctx, sliceValue), "")
			}
		case *types.Pointer:
			otherType := expr.Type().(*types.Basic)
			if otherType.Kind() == types.UnsafePointer {
				value.LLVMValueRef = llvm.BuildPointerCast(c.builder, fromValue.UnderlyingValue(ctx), typeTo.valueType, "")
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
					value.LLVMValueRef = llvm.GetParam(fn.value, uint(i))
				}
			}
		} else {
			panic("function does not exist")
		}
	case *ssa.ChangeInterface:
		panic("not implemented")
	case *ssa.ChangeType:
		value = c.createExpression(ctx, expr.X)
	case *ssa.Extract:
		// Get the return struct (tuple)
		structValue := c.createExpression(ctx, expr.Tuple).UnderlyingValue(ctx)
		value.LLVMValueRef = llvm.BuildExtractValue(c.builder, structValue, uint(expr.Index), "")
	case *ssa.Field:
		structValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		value.LLVMValueRef = llvm.BuildExtractValue(c.builder, structValue, uint(expr.Field), "")
	case *ssa.FieldAddr:
		var structType llvm.LLVMTypeRef
		structValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)

		pointerType := expr.X.Type().Underlying().(*types.Pointer)

		// Nil check the pointer
		c.createRuntimeCall(ctx, "nilCheck", []llvm.LLVMValueRef{structValue})
		structType = c.createType(ctx, pointerType.Elem()).valueType

		// Create a GEP to get the address of the field within the struct
		value.LLVMValueRef = llvm.BuildStructGEP2(
			c.builder,
			structType,
			structValue,
			uint(expr.Field), "")
	case *ssa.FreeVar:
		fnObj := c.functions[expr.Parent()]
		varType := c.createType(ctx, expr.Type())

		// The last parameter of the function is the struct containing free var values
		state := llvm.GetParam(fnObj.value, llvm.CountParamTypes(fnObj.llvmType)-1)

		// FreeVars are stored in the state struct in the order that they appear in ssa.Function
		for i, freeVar := range fnObj.def.FreeVars {
			if freeVar == expr {
				// Get the address of the value to be loaded from the struct
				addr := llvm.BuildStructGEP2(c.builder, fnObj.stateType, state, uint(i), "")

				// Load the value from the struct
				value.LLVMValueRef = llvm.BuildLoad2(c.builder, varType.valueType, addr, "")

				// Stop
				break
			}
		}
	case *ssa.Function:
		// Check the function cache first
		fn, ok := c.functions[expr]
		if !ok {
			fn = c.createFunction(ctx, expr)
			c.functions[expr] = fn
		}

		// Return a pointer to the function
		value.LLVMValueRef = fn.value
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
			value.LLVMValueRef = llvm.GetNamedGlobal(c.module, value.linkname)
			if value.LLVMValueRef == nil {
				// Look for a function with this name
				value.LLVMValueRef = llvm.GetNamedFunction(c.module, value.linkname)
				if value.LLVMValueRef != nil {
					value.LLVMValueRef = llvm.BuildBitCast(c.builder, value, globalType.valueType, "")
					value.LLVMValueRef = llvm.BuildPointerCast(c.builder, value, globalType.valueType, "")
				} else {
					// Create a global with external linkage to some variable with the specified link name.
					value.LLVMValueRef = llvm.AddGlobal(c.module, globalType.valueType, value.linkname)
					llvm.SetLinkage(value, llvm.LLVMLinkage(llvm.ExternalLinkage))
				}
			}
		} else if expr.Pkg != c.currentPackage(ctx) {
			// Create a extern global value
			value.LLVMValueRef = llvm.AddGlobal(c.module, globalType.valueType, value.linkname)
			llvm.SetLinkage(value, llvm.LLVMLinkage(llvm.ExternalLinkage))
		} else {
			value.LLVMValueRef = c.createGlobalValue(ctx, llvm.ConstNull(globalType.valueType), value.linkname)
			if !value.exported {
				llvm.SetLinkage(value, llvm.LLVMLinkage(llvm.PrivateLinkage))
			}
		}
	case *ssa.Index:
		arrayValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		indexValue := c.createExpression(ctx, expr.Index).UnderlyingValue(ctx)

		// Get the address of the element at the index within the array
		var addr llvm.LLVMTypeRef
		var loadElementType llvm.LLVMTypeRef

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())
			loadElementType = elementType.valueType

			// Create a runtime call to retrieve the address of the element at index I
			addr = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(ctx, arrayValue),
				indexValue,
				c.createTypeDescriptor(ctx, elementType),
			})
		case *types.Basic: // Strings
			loadElementType = llvm.Int8TypeInContext(c.currentContext(ctx))

			// Create a runtime call to retrieve the address of the element at index I
			addr = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(ctx, arrayValue),
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
		value.LLVMValueRef = llvm.BuildLoad2(c.builder, loadElementType, addr, "")
	case *ssa.IndexAddr:
		arrayValue := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		indexValue := c.createExpression(ctx, expr.Index).UnderlyingValue(ctx)
		indexType := expr.Index.Type().Underlying().(*types.Basic)

		switch operandType := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			// Get the element type of the slice
			elementType := c.createType(ctx, operandType.Elem())

			// Create a runtime call to retrieve the address of the element at index I
			value.LLVMValueRef = c.createRuntimeCall(ctx, "sliceIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(ctx, arrayValue),
				llvm.BuildIntCast2(c.builder,
					indexValue,
					llvm.Int32TypeInContext(c.currentContext(ctx)),
					indexType.Info()&types.IsUnsigned == 0,
					""),
				c.createTypeDescriptor(ctx, elementType),
			})
		case *types.Basic: // Strings
			// Create a runtime call to retrieve the address of the element at index I
			value.LLVMValueRef = c.createRuntimeCall(ctx, "stringIndexAddr", []llvm.LLVMValueRef{
				c.addressOf(ctx, arrayValue),
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
			value.LLVMValueRef = llvm.BuildGEP2(c.builder, llvm.GetElementType(arrayType), arrayValue,
				[]llvm.LLVMValueRef{indexValue}, "")
		}
	case *ssa.Lookup:
		panic("not implemented")
	case *ssa.MakeChan:
		panic("not implemented")
	case *ssa.MakeClosure:
		fnObj := c.functions[expr.Fn.(*ssa.Function)]

		// Create the state struct
		value.LLVMValueRef = c.createAlloca(ctx, fnObj.stateType, "")
		for i, bound := range expr.Bindings {
			val := c.createExpression(ctx, bound).UnderlyingValue(ctx)
			addr := llvm.BuildStructGEP2(c.builder, fnObj.stateType, value, uint(i), "")
			llvm.BuildStore(c.builder, val, addr)
		}
	case *ssa.MakeInterface:
		value.LLVMValueRef = c.makeInterface(ctx, expr.X)
	case *ssa.MakeSlice:
		lenValue := c.createExpression(ctx, expr.Len).UnderlyingValue(ctx)
		lenValueType := expr.Len.Type().Underlying().(*types.Basic)
		capValue := c.createExpression(ctx, expr.Cap).UnderlyingValue(ctx)
		capValueType := expr.Cap.Type().Underlying().(*types.Basic)
		elementType := c.createType(ctx, expr.Type().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)

		// Create the runtime call
		var slice llvm.LLVMValueRef
		slice = c.createRuntimeCall(ctx, "sliceMake", []llvm.LLVMValueRef{
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

		// Load the slice value
		//sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")
		value.LLVMValueRef = slice //llvm.BuildLoad2(c.builder, sliceType, c.addressOf(ctx, slice), "")
	case *ssa.MultiConvert:
		panic("not implemented")
	case *ssa.Next:
		iter := c.createExpression(ctx, expr.Iter).UnderlyingValue(ctx)

		// Create the respective runtime call
		if expr.IsString {
			value.LLVMValueRef = c.createRuntimeCall(ctx, "stringRange", []llvm.LLVMValueRef{iter})
		} else {
			panic("map type not implemented")
		}
	case *ssa.Phi:
		phiType := c.createType(ctx, expr.Type())

		// Build the Phi node operator
		value.LLVMValueRef = llvm.BuildPhi(c.builder, phiType.valueType, "")

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
		llvm.AddIncoming(value.LLVMValueRef, edgeValues.Ref(ctx), blocks)
	case *ssa.Range:
		containerVal := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		switch expr.X.Type().Underlying().(type) {
		case *types.Basic:
			// Create a struct representing the string/map iterator
			strType := llvm.StructTypeInContext(c.currentContext(ctx), []llvm.LLVMTypeRef{
				c.createType(ctx, expr.X.Type().Underlying()).valueType,
				c.int32Type(ctx),
			}, false)
			value.LLVMValueRef = c.createAlloca(ctx, strType, "string_iterator")

			// Set the range parameter values in the struct
			llvm.BuildStore(c.builder, containerVal, llvm.BuildStructGEP2(c.builder, strType, value, 0, "str_addr"))
			llvm.BuildStore(c.builder, llvm.ConstInt(c.int64Type(ctx), 0, false), llvm.BuildStructGEP2(c.builder, strType, value, 1, "index_arr"))
		case *types.Map:
			panic("map type not implemented")
		}
	case *ssa.Slice:
		var low, high, max llvm.LLVMValueRef
		if expr.Low != nil {
			low = c.createExpression(ctx, expr.Low).UnderlyingValue(ctx)
		}

		if expr.High != nil {
			high = c.createExpression(ctx, expr.High).UnderlyingValue(ctx)
		}

		if expr.Max != nil {
			max = c.createExpression(ctx, expr.Max).UnderlyingValue(ctx)
		}

		array := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)

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
		x := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
		newType := c.createType(ctx, expr.AssertedType)

		// The value needs to be passed by address. Create an alloca for it and store the value
		xAlloca := c.addressOf(ctx, x)
		xAlloca = llvm.BuildBitCast(c.builder, xAlloca, c.ptrType.valueType, "")

		// create the runtime call
		result := c.createRuntimeCall(ctx, "typeAssert", []llvm.LLVMValueRef{
			xAlloca,
			c.createTypeDescriptor(ctx, c.createType(ctx, expr.X.Type())),
			c.createTypeDescriptor(ctx, newType),
		})

		// Get addresses of return value
		result = c.addressOf(ctx, result)
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
		value.LLVMValueRef = c.createUnOp(ctx, expr)
	default:
		panic("unimplemented expression type: " + expr.String())
	}

	if value.LLVMValueRef == nil {
		panic("nil value")
	}

	// Cache the value
	c.values[expr] = value

	return value
}
