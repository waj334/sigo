package ssa

import (
	"context"
	"go/ast"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitCallExpr(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	tv := b.config.Info.Types[expr.Fun]

	if F, ok := b.objectOf(expr.Fun).(*types.Func); ok {
		// Set up type inference.
		signature := F.Type().(*types.Signature)
		ctx = newContextWithLhsList(ctx, tupleTypes(signature.Params()))
	}

	if tv.IsBuiltin() {
		// Emit the respective runtime call.
		return b.emitBuiltinCall(ctx, expr)
	} else if b.isIntrinsic(expr) {
		return b.emitIntrinsic(ctx, expr)
	} else if tv.IsType() {
		// Evaluate the value to convert.
		X := b.emitExpr(ctx, expr.Args[0])[0]
		// Perform type conversion.
		srcType := b.typeOf(expr.Args[0])
		destType := tv.Type
		value := b.emitTypeConversion(ctx, X, srcType, destType, location)
		return []mlir.Value{value}
	} else {
		// Evaluate call arguments.
		argValues := make([]mlir.Value, len(expr.Args))
		for i, expr := range expr.Args {
			ctx = newContextWithRhsIndex(ctx, i)
			argValues[i] = b.emitExpr(ctx, expr)[0]
			switch expr := expr.(type) {
			case *ast.Ident:
				if expr.Obj != nil {
					if _, ok := expr.Obj.Decl.(*ast.FuncDecl); ok {
						// Only create a function struct value if the identifier is that of a function declaration.
						if typeIs[*types.Signature](b.typeOf(expr)) {
							argValues[i] = b.createFunctionValue(ctx, argValues[i], nil, location)
						}
					}
				}
			}
		}

		signature := baseType(b.typeOf(expr.Fun)).(*types.Signature)

		// Handle interface arguments.
		for i := range expr.Args {
			argT := b.typeOf(expr.Args[i])
			paramT := signature.Params().At(min(i, signature.Params().Len()-1)).Type()

			switch baseType(paramT).(type) {
			case *types.Interface:
				if !isNil(argT) && !types.Identical(paramT, argT) {
					if types.IsInterface(baseType(argT)) {
						// Convert from interface A to interface B.
						argValues[i] = b.emitChangeType(ctx, paramT, argValues[i], location)
					} else {
						// Generate methods for named types.
						if T, ok := argT.(*types.Named); ok {
							b.queueNamedTypeJobs(ctx, T)
						}

						// Create an interface value from the value expression.
						argValues[i] = b.emitInterfaceValue(ctx, paramT, argValues[i], location)
					}
				}
			}
		}

		switch Fun := expr.Fun.(type) {
		case *ast.SelectorExpr:
			T := b.typeOf(Fun.X)
			funcObj := b.objectOf(Fun.Sel).(*types.Func)
			switch T.Underlying().(type) {
			case *types.Interface:
				// Evaluate the interface value.
				X := b.emitExpr(ctx, Fun.X)[0]

				// Emit the interface call.
				return b.emitInterfaceCall(ctx, X, funcObj, argValues, location)
			default:
				signature := funcObj.Type().(*types.Signature)
				if signature.Recv() != nil {
					recvType := signature.Recv().Type()
					//recvT := b.GetStoredType(ctx, recvType)
					actualRecvType := b.typeOf(Fun.X)

					recvValue := b.NewTempValue(b.addressOf(ctx, Fun.X, location))

					var recv mlir.Value
					if isPointer(b.typeOf(Fun.X)) {
						// Load the address stored in the value.
						recv = recvValue.Load(ctx, location)
					} else {
						// Get the address of the value.
						recv = recvValue.Pointer(ctx, location)
					}

					// The callee is from an embedded type if it matches a receiver from any method of a type embedded
					// in a struct.
					actualStructType := baseStructTypeOf(actualRecvType)
					if actualStructType != nil {
						embeddedType := recvType
						if ptrType, ok := recvType.(*types.Pointer); ok {
							embeddedType = ptrType.Elem()
						}

						for i := 0; i < actualStructType.NumFields(); i++ {
							field := actualStructType.Field(i)
							if field.Embedded() {
								matches := false
								load := false
								switch fieldType := field.Type().(type) {
								case *types.Pointer:
									matches = types.Identical(embeddedType, fieldType.Elem())
									load = true
								default:
									matches = types.Identical(embeddedType, fieldType)
								}

								if matches {
									ptrT := b.pointerOf(ctx, field.Type())
									structT := b.GetStoredType(ctx, actualStructType)
									gepOp := mlir.GoCreateGepOperation2(b.ctx, recv, structT, []any{0, i}, ptrT, location)
									appendOperation(ctx, gepOp)
									recv = resultOf(gepOp)
									if load {
										recvValue = b.NewTempValue(recv)
										recv = recvValue.Load(ctx, location)
									}
									break
								}
							}
						}
					}

					if !isPointer(recvType) {
						recvValue = b.NewTempValue(recv)
						recv = recvValue.Load(ctx, location)
					}

					// Prepend the receiver value to the call args.
					argValues = append([]mlir.Value{recv}, argValues...)
				}
				return b.emitGeneralCall(ctx, Fun.Sel, funcObj, argValues, location)
			}
		case *ast.FuncLit:
			// Evaluate the function literal symbol to call indirectly.
			fnValue := b.emitExpr(ctx, expr.Fun)[0]
			argValues = b.emitVariadicArgs(ctx, signature, argValues, location)
			enclosingFuncData := currentFuncData(ctx)

			// Get the data for the function literal.
			data := enclosingFuncData.anonymousFuncs[Fun]
			data.mutex.RLock()
			if len(data.freeVars) > 0 {
				// Create the context struct for the call.
				ctxValue, ctxType := data.createContextStructValue(ctx, b, location)

				// Allocate stack to store the context struct value.
				allocaOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(ctxType), ctxType, nil, false, location)
				appendOperation(ctx, allocaOp)

				// Store the argument pack at the address.
				storeOp := mlir.GoCreateStoreOperation(b.ctx, ctxValue, resultOf(allocaOp), location)
				appendOperation(ctx, storeOp)

				// Append the context struct value to the arg list.
				argValues = append(argValues, b.bitcastTo(ctx, resultOf(allocaOp), b.ptr, location))
			}

			// Gather call result types from its signature.
			resultTypes := make([]mlir.Type, mlir.FunctionTypeGetNumResults(data.mlirType))
			for i := range resultTypes {
				resultTypes[i] = mlir.FunctionTypeGetResult(data.mlirType, i)
			}
			data.mutex.RUnlock()

			// Emit the indirect call.
			op := mlir.GoCreateCallIndirectOperation(b.ctx, fnValue, resultTypes, argValues, location)
			appendOperation(ctx, op)
			return resultsOf(op)
		default:
			switch funcObj := b.objectOf(Fun).(type) {
			case *types.Func:
				return b.emitGeneralCall(ctx, Fun.(*ast.Ident), funcObj, argValues, location)
			case *types.Var:
				// Evaluate the func value.
				fnValue := b.emitExpr(ctx, expr.Fun)[0]

				signature := baseType(funcObj.Type()).(*types.Signature)
				argValues = b.emitVariadicArgs(ctx, signature, argValues, location)

				// Gather parameter types.
				// NOTE: This starts with zero length so that the environment pointer can be appended even when there's
				//       no parameters.
				paramTypes := make([]mlir.Type, 0, signature.Params().Len())
				for i := 0; i < signature.Params().Len(); i++ {
					paramTypes = append(paramTypes, b.GetStoredType(ctx, signature.Params().At(i).Type()))
				}

				// Gather call result types from its signature.
				resultTypes := make([]mlir.Type, signature.Results().Len())
				for i := 0; i < signature.Results().Len(); i++ {
					resultTypes[i] = b.GetStoredType(ctx, signature.Results().At(i).Type())
				}

				// Create the closure function type
				closureFnType := mlir.FunctionTypeGet(b.ctx, append(paramTypes, b.ptr), resultTypes)

				// Extract the callee ptr from the func value struct.
				extractOp := mlir.GoCreateExtractOperation(b.ctx, 0, b.ptr, fnValue, location)
				appendOperation(ctx, extractOp)
				funcPtr := resultOf(extractOp)
				funcPtr = b.bitcastTo(ctx, funcPtr, closureFnType, location)

				// Extract the environment pointer from the func value struct.
				extractOp = mlir.GoCreateExtractOperation(b.ctx, 1, b.ptr, fnValue, location)
				appendOperation(ctx, extractOp)
				envPtr := resultOf(extractOp)

				// Emit the indirect call.
				op := mlir.GoCreateCallIndirectOperation(b.ctx, funcPtr, resultTypes, append(argValues, envPtr), location)
				appendOperation(ctx, op)
				return resultsOf(op)
			default:
				panic("unhandled")
			}
		}
	}
}

func (b *Builder) emitGeneralCall(ctx context.Context, ident *ast.Ident, obj *types.Func, args []mlir.Value, location mlir.Location) []mlir.Value {
	callee := qualifiedFuncName(obj)
	signature := obj.Type().Underlying().(*types.Signature)

	if signature.Recv() != nil {
		// Get the name of the method receiver's named type.
		var typename string
		if isPointer(signature.Recv().Type()) {
			typename = signature.Recv().Type().(*types.Pointer).Elem().(*types.Named).Obj().Name()
		} else {
			typename = signature.Recv().Type().(*types.Named).Obj().Name()
		}

		// Format the callee.
		callee = qualifiedName(typename+"."+obj.Name(), obj.Pkg())
	}

	// Is the callee a generic function?
	if signature.TypeParams().Len() > 0 || signature.RecvTypeParams().Len() > 0 {
		// Need to instantiate this generic function.
		data := b.genericFuncs[callee]
		instance := b.config.Info.Instances[ident]
		instanceData := b.createFuncInstance(ctx, signature, instance, data)
		callee = instanceData.symbol
		signature = instanceData.signature
	}

	args = b.emitVariadicArgs(ctx, signature, args, location)

	// Gather call result types from its signature.
	resultTypes := make([]mlir.Type, signature.Results().Len())
	for i := 0; i < signature.Results().Len(); i++ {
		resultTypes[i] = b.GetStoredType(ctx, signature.Results().At(i).Type())
	}

	// Create the function call.
	symbol := b.resolveSymbol(callee)
	b.queueJob(ctx, symbol)

	callOp := mlir.GoCreateCallOperation(b.ctx, symbol, resultTypes, args, location)
	appendOperation(ctx, callOp)
	return resultsOf(callOp)
}

func (b *Builder) emitInterfaceCall(ctx context.Context, X mlir.Value, obj *types.Func, args []mlir.Value, location mlir.Location) []mlir.Value {
	method := obj.Name()
	T := b.createSignatureType(ctx, obj.Type().(*types.Signature), false)
	args = b.emitVariadicArgs(ctx, obj.Type().(*types.Signature), args, location)

	// Create the interface call operation
	op := mlir.GoCreateInterfaceCall(b.ctx, method, T, X, args, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitGoStatement(ctx context.Context, stmt *ast.GoStmt) {
	location := b.location(stmt.Pos())

	// Evaluate the call arguments
	callArgs := b.exprValues(ctx, stmt.Call.Args...)

	// Evaluate the function based on the statement's callee expression.
	switch Fun := stmt.Call.Fun.(type) {
	case *ast.FuncLit:
		// Emit the literal function.
		F := b.emitExpr(ctx, Fun)[0]

		// Get the data for this function.
		enclosingFuncData := currentFuncData(ctx)
		data := enclosingFuncData.anonymousFuncs[Fun]

		if len(data.freeVars) > 0 {
			// Create the context struct value.
			contextValue, contextType := data.createContextStructValue(ctx, b, location)

			// Allocate heap for the context value.
			allocOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, contextType, nil, true, location)
			appendOperation(ctx, allocOp)

			// Store the context value at the heap address.
			storeOp := mlir.GoCreateStoreOperation(b.ctx, contextValue, resultOf(allocOp), location)
			appendOperation(ctx, storeOp)

			// Prepend the context value to the call args.
			callArgs = append([]mlir.Value{resultOf(allocOp)}, callArgs...)
		}

		// Emit the goroutine operation.
		op := mlir.GoCreateGoOperation(b.ctx, F, nil, callArgs, location)
		appendOperation(ctx, op)
	case *ast.Ident:
		// Evaluate the callee.
		F := b.emitExpr(ctx, Fun)[0]

		// Emit the goroutine operation.
		op := mlir.GoCreateGoOperation(b.ctx, F, nil, callArgs, location)
		appendOperation(ctx, op)
	case *ast.SelectorExpr:
		T := b.typeOf(Fun.X)
		switch T.Underlying().(type) {
		case *types.Interface:
			// Evaluate the interface value.
			F := b.emitExpr(ctx, Fun.X)[0]

			// Emit the goroutine operation.
			op := mlir.GoCreateGoOperation(b.ctx, F, mlir.StringAttrGet(b.ctx, Fun.Sel.Name), callArgs, location)
			appendOperation(ctx, op)
			return
		default:
			// This is a named type method or a package function.
			// Evaluate the callee.
			F := b.emitExpr(ctx, Fun)[0]

			switch b.objectOf(Fun.X).(type) {
			case *types.Var:
				// Evaluate the receiver value.
				recv := b.emitExpr(ctx, Fun.X)[0]

				// Prepend the receiver value to the call args.
				callArgs = append([]mlir.Value{recv}, callArgs...)
			}

			// Emit the goroutine operation.
			op := mlir.GoCreateGoOperation(b.ctx, F, nil, callArgs, location)
			appendOperation(ctx, op)
		}
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitDeferStatement(ctx context.Context, stmt *ast.DeferStmt) {
	location := b.location(stmt.Pos())

	// Evaluate the call arguments
	callArgs := b.exprValues(ctx, stmt.Call.Args...)

	// Evaluate the function based on the statement's callee expression.
	switch Fun := stmt.Call.Fun.(type) {
	case *ast.FuncLit:
		// Emit the literal function.
		F := b.emitExpr(ctx, Fun)[0]

		// Get the data for this function.
		enclosingFuncData := currentFuncData(ctx)
		data := enclosingFuncData.anonymousFuncs[Fun]

		if len(data.freeVars) > 0 {
			// Create the context struct value.
			contextValue, contextType := data.createContextStructValue(ctx, b, location)

			// Allocate heap for the context value.
			allocOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, contextType, nil, true, location)
			appendOperation(ctx, allocOp)

			// Store the context value at the heap address.
			storeOp := mlir.GoCreateStoreOperation(b.ctx, contextValue, resultOf(allocOp), location)
			appendOperation(ctx, storeOp)

			// Prepend the context value to the call args.
			callArgs = append([]mlir.Value{resultOf(allocOp)}, callArgs...)
		}

		// Emit the goroutine operation.
		op := mlir.GoCreateDeferOperation(b.ctx, F, nil, callArgs, location)
		appendOperation(ctx, op)
	case *ast.Ident:
		// Evaluate the callee.
		F := b.emitExpr(ctx, Fun)[0]

		// Emit the goroutine operation.
		op := mlir.GoCreateDeferOperation(b.ctx, F, nil, callArgs, location)
		appendOperation(ctx, op)
	case *ast.SelectorExpr:
		T := b.typeOf(Fun.X)

		switch T.Underlying().(type) {
		case *types.Interface:
			// Evaluate the interface value.
			F := b.emitExpr(ctx, Fun.X)[0]

			// Emit the goroutine operation.
			op := mlir.GoCreateDeferOperation(b.ctx, F, mlir.StringAttrGet(b.ctx, Fun.Sel.Name), callArgs, location)
			appendOperation(ctx, op)
			return
		default:
			// Evaluate the callee.
			F := b.emitExpr(ctx, Fun)[0]

			switch b.objectOf(Fun.X).(type) {
			case *types.Var:
				// Evaluate the receiver value.
				recv := b.emitExpr(ctx, Fun.X)[0]

				// Prepend the receiver value to the call args.
				callArgs = append([]mlir.Value{recv}, callArgs...)
			}

			// Emit the goroutine operation.
			op := mlir.GoCreateDeferOperation(b.ctx, F, nil, callArgs, location)
			appendOperation(ctx, op)
		}
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitVariadicArgs(ctx context.Context, signature *types.Signature, args []mlir.Value, location mlir.Location) []mlir.Value {
	if signature.Variadic() {
		variadicBegin := signature.Params().Len() - 1
		numVariadicArgs := len(args) - variadicBegin
		variadicArgType := signature.Params().At(variadicBegin).Type().(*types.Slice)
		elementType := b.GetStoredType(ctx, variadicArgType.Elem())

		if mlir.TypeEqual(mlir.ValueGetType(args[variadicBegin]), b.GetStoredType(ctx, variadicArgType)) {
			// This is ellipsis (...).
			return args
		}

		// Create the backing array for the slice that will contain the variadic arguments.
		constLength := b.emitConstInt(ctx, int64(numVariadicArgs), b.si, location)
		allocaOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, elementType, constLength, false, location)
		appendOperation(ctx, allocaOp)
		mlir.OperationMoveBefore(mlir.ValueGetDefiningOperation(constLength), allocaOp)

		// Fill the backing array.
		for i, arg := range args[variadicBegin:] {
			// Gep into the backing array to the position where the current argument should be stored.
			gepOp := mlir.GoCreateGepOperation2(b.ctx, resultOf(allocaOp), b._any, []any{i}, mlir.GoCreatePointerType(elementType), location)
			appendOperation(ctx, gepOp)

			// Store the argument value.
			storeOp := mlir.GoCreateStoreOperation(b.ctx, arg, resultOf(gepOp), location)
			appendOperation(ctx, storeOp)
		}

		// Create the variadic argument slice.
		varArg := b.emitConstSlice(ctx, resultOf(allocaOp), numVariadicArgs, location)

		// Reinterpret the runtime slice as the dialect's equivalent.
		varArg = b.bitcastTo(ctx, varArg, b.GetStoredType(ctx, variadicArgType), location)

		// Slice the input arguments to remove the individual variadic arguments and append the variadic argument slice
		// to it.
		args = append(args[:variadicBegin], varArg)
	}
	return args
}

func (b *Builder) createInterfaceCallWrapper(ctx context.Context, symbol string, callee string, iface *types.Interface, signature *types.Signature, argTypes []mlir.Type) {
	b.thunkMutex.Lock()
	defer b.thunkMutex.Unlock()

	// Look up the thunk in the symbol table first.
	if _, ok := b.thunks[symbol]; !ok {
		// Prepend the interface type to the beginning of the argument pack type list.
		argTypes = append([]mlir.Type{b.GetStoredType(ctx, iface)}, argTypes...)

		// Create the argument struct type.
		argPackType := mlir.GoCreateLiteralStructType(b.ctx, argTypes)

		// Any argument excluded from the argument pack MUST be passed to the resulting thunk directly.
		// NOTE: The interface value is added to the parameter count.
		paramTypes := []mlir.Type{mlir.GoCreatePointerType(argPackType)}
		for i := len(argTypes); i < signature.Params().Len()+1; i++ {
			paramTypes = append(paramTypes, b.GetStoredType(ctx, signature.Params().At(i).Type()))
		}
		paramLocs := make([]mlir.Location, len(paramTypes))
		fill(paramLocs, b._noLoc)

		// Collect the result types.
		resultTypes := make([]mlir.Type, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			resultTypes = append(resultTypes, b.GetStoredType(ctx, signature.Results().At(i).Type()))
		}

		// Create thunk to wrap the method call.
		region := mlir.RegionCreate()
		ctx = newContextWithRegion(ctx, region)

		entryBlock := mlir.BlockCreate2(paramTypes, paramLocs)
		mlir.RegionAppendOwnedBlock(region, entryBlock)
		buildBlock(ctx, entryBlock, func() {
			argPackPtrValue := mlir.BlockGetArgument(entryBlock, 0)
			args := b.unpackArgPack(ctx, argTypes, argPackPtrValue, b._noLoc)

			// Gather the remaining arguments
			for i := 1; i < mlir.BlockGetNumArguments(entryBlock); i++ {
				args = append(args, mlir.BlockGetArgument(entryBlock, i))
			}

			// Call the method.
			signatureType := b.createSignatureType(ctx, signature, true)
			callOp := mlir.GoCreateInterfaceCall(b.ctx, callee, signatureType, args[0], args[1:], b._noLoc)
			appendOperation(ctx, callOp)

			// Return the results.
			returnOp := mlir.GoCreateReturnOperation(b.ctx, resultsOf(callOp), b._noLoc)
			appendOperation(ctx, returnOp)
		})

		// Create the function operation for this thunk.
		thunkFuncType := mlir.FunctionTypeGet(b.ctx, paramTypes, resultTypes)
		state := mlir.OperationStateGet("func.func", b._noLoc)
		mlir.OperationStateAddOwnedRegions(state, []mlir.Region{region})
		mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
			b.namedOf("function_type", mlir.TypeAttrGet(thunkFuncType)),
			b.namedOf("sym_name", mlir.StringAttrGet(b.config.Ctx, symbol)),
			b.namedOf("sym_visibility", mlir.StringAttrGet(b.config.Ctx, "private")),
		})

		funcOp := mlir.OperationCreate(state)

		// This operation will be added later safely.
		b.addToModuleMutex.Lock()
		b.addToModule[symbol] = funcOp
		b.addToModuleMutex.Unlock()

		b.thunks[symbol] = struct{}{}
	}
}
