package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type calleeType int

const (
	calleeIsSymbol calleeType = iota
	calleeIsClosure
	calleeIsInterface
)

type callOpArgs struct {
	calleeType calleeType
	function   string
	callee     mlir.ValueLike
	args       []mlir.ValueLike
	results    []mlir.TypeLike
	expr       *ast.CallExpr
	load       bool
	typeMap    TypeParamMap
	signature  *types.Signature
}

func (b *Builder) extractCallOpArgs(ctx context.Context, expr *ast.CallExpr) callOpArgs {
	var signature *types.Signature
	var call callOpArgs
	instanceResolved := false

	location := b.location(ctx, expr.Pos())
	info := currentInfo(ctx)
	funcObj := b.objectOf(ctx, expr.Fun)
	calleeExpr := expr.Fun

	for {
		// Determine what the callee is.
		switch Fun := calleeExpr.(type) {
		case *ast.Ident:
			switch obj := funcObj.(type) {
			case *types.Func:
				call.calleeType = calleeIsSymbol
				call.function = qualifiedFuncName(obj)
				signature = baseType(obj.Type()).(*types.Signature)
			case *types.Var:
				call.calleeType = calleeIsClosure
				call.callee = b.emitExpr(ctx, Fun)[0]
				signature = baseType(obj.Type()).(*types.Signature)
			default:
				panic("unhandled")
			}
		case *ast.FuncLit:
			call.calleeType = calleeIsClosure
			call.callee = b.emitExpr(ctx, Fun)[0]
			signature = b.typeOf(ctx, Fun).(*types.Signature)
		case *ast.SelectorExpr:
			sel := info.Selections[Fun]
			signature = baseType(funcObj.Type()).(*types.Signature)

			if sel != nil {
				switch sel.Kind() {
				case types.FieldVal:
					call.calleeType = calleeIsClosure
					call.callee = b.emitExpr(ctx, Fun)[0]
				case types.MethodVal, types.MethodExpr:
					funcObj := funcObj.(*types.Func)
					recvT := sel.Recv()
					origRecvT := recvT

					// Resolve any nested TypeParams in the receiver type through
					// the enclosing function's type map. This handles cases like
					// *uniqueMap[T] → *uniqueMap[addrDetail] where T is a TypeParam
					// nested inside a Pointer/Named composite type.
					if outerTypeMap := currentTypeMap(ctx); outerTypeMap != nil && containsTypeParam(recvT) {
						recvT = resolveTypeInTypeMap(recvT, outerTypeMap)
					}

					if typeParam, ok := recvT.(*types.TypeParam); ok {
						recvT = resolveType(ctx, typeParam)
					}

					// If the receiver was resolved from a TypeParam to a concrete
					// Named type, find the matching method on that type instead of
					// using the interface constraint's method.
					if recvT != origRecvT {
						// Dereference pointer if present to get the named type.
						lookupT := types.Unalias(recvT)
						if ptr, ok := lookupT.(*types.Pointer); ok {
							lookupT = types.Unalias(ptr.Elem())
						}
						if namedRecvT, ok := lookupT.(*types.Named); ok && !types.IsInterface(namedRecvT) {
							// Use types.LookupFieldOrMethod to find the method,
							// which handles pointer receivers and promoted methods.
							obj, _, _ := types.LookupFieldOrMethod(recvT, true, namedRecvT.Obj().Pkg(), funcObj.Name())
							if method, ok := obj.(*types.Func); ok {
								funcObj = method
							}
						}
					}

					// Update the signature.
					signature = funcObj.Signature()

					if types.IsInterface(types.Unalias(recvT)) {
						call.calleeType = calleeIsInterface
						call.callee = b.emitExpr(ctx, Fun.X)[0]
						call.function = funcObj.Name()
					} else {
						call.calleeType = calleeIsSymbol
						call.function = qualifiedFuncName(funcObj)

						// Check if this is a method on an instantiated generic named type.
						// If so, resolve directly to the concrete function instance so that
						// the generic check at the end of the loop is skipped entirely.
						//
						// First try the expression's receiver type. If that doesn't have
						// TypeArgs, fall back to the method's signature receiver — this
						// handles promoted methods from embedded generic types (e.g.,
						// uniqueMap embeds *canonMap[T], so the expression receiver is
						// uniqueMap but the method's actual receiver is canonMap[T]).
						var namedRecv *types.Named
						if nr, ok := namedRecvType(recvT); ok && nr.TypeArgs().Len() > 0 {
							namedRecv = nr
						} else if sigRecv := funcObj.Type().(*types.Signature).Recv(); sigRecv != nil {
							if nr, ok := namedRecvType(sigRecv.Type()); ok && nr.TypeArgs().Len() > 0 {
								namedRecv = nr
							}
						}

						if namedRecv != nil {
							origin := namedRecv.Origin()
							typeMap := make(TypeParamMap)
							allConcrete := true
							outerTypeMap := currentTypeMap(ctx)
							for i := 0; i < origin.TypeParams().Len(); i++ {
								targ := namedRecv.TypeArgs().At(i)
								// Resolve any TypeParams through the enclosing function's type map.
								if containsTypeParam(targ) {
									if outerTypeMap != nil {
										targ = resolveTypeInTypeMap(targ, outerTypeMap)
										if containsTypeParam(targ) {
											allConcrete = false
											break
										}
									} else {
										allConcrete = false
										break
									}
								}
								typeMap[origin.TypeParams().At(i).Index()] = targ
							}

							if allConcrete {
								// The receiver is fully instantiated, so the signature's
								// params/results are already concrete. Mark as resolved to
								// prevent the generic instantiation block from triggering
								// on the misleading RecvTypeParams.
								instanceResolved = true

								// Find the matching method on the origin (uninstantiated) type
								// so we can look it up in genericFuncs under its generic symbol.
								//
								// Note: origin.NumMethods() only returns EXPLICIT (non-promoted)
								// methods. For promoted methods from embedded generic types (e.g.,
								// uniqueMap embeds *canonMap[T], so LoadOrStore is promoted from
								// canonMap), we need to trace through the method's actual receiver
								// type to find the declaring type.
								var originMethod *types.Func
								for i := 0; i < origin.NumMethods(); i++ {
									if origin.Method(i).Name() == funcObj.Name() {
										originMethod = origin.Method(i)
										break
									}
								}

								if originMethod == nil {
									// Promoted from an embedded generic type. Trace through the
									// method's actual receiver type to find the declaring generic type.
									sigRecvT := funcObj.Type().(*types.Signature).Recv().Type()
									if ptr, ok := sigRecvT.(*types.Pointer); ok {
										sigRecvT = ptr.Elem()
									}
									// Resolve TypeParams using the typeMap from the outer receiver
									// (e.g., uniqueMap[addrDetail]'s T0 → addrDetail).
									resolvedRecvT := resolveTypeInTypeMap(sigRecvT, typeMap)
									if embeddedNamed, ok := resolvedRecvT.(*types.Named); ok && embeddedNamed.TypeArgs().Len() > 0 {
										embeddedOrigin := embeddedNamed.Origin()
										for i := 0; i < embeddedOrigin.NumMethods(); i++ {
											if embeddedOrigin.Method(i).Name() == funcObj.Name() {
												originMethod = embeddedOrigin.Method(i)
												break
											}
										}
										if originMethod != nil {
											// Use the concrete signature from the instantiated embedded
											// type so that createFuncInstance stores correct types.
											for j := 0; j < embeddedNamed.NumMethods(); j++ {
												if embeddedNamed.Method(j).Name() == funcObj.Name() {
													signature = embeddedNamed.Method(j).Type().(*types.Signature)
													break
												}
											}
											// Rebuild typeMap from the embedded type's TypeParams/TypeArgs.
											typeMap = make(TypeParamMap)
											for j := 0; j < embeddedOrigin.TypeParams().Len(); j++ {
												typeMap[embeddedOrigin.TypeParams().At(j).Index()] = embeddedNamed.TypeArgs().At(j)
											}
										}
									}
								}

								if originMethod != nil {
									genericSymbol := qualifiedFuncName(originMethod)

									// Ensure the generic method is registered. It may not
									// have been processed yet if the declaring type's methods
									// are discovered through embedding during instance emission.
									b.queueJob(ctx, genericSymbol)

									b.funcDeclDataMutex.RLock()
									genericData, ok := b.genericFuncs[genericSymbol]
									b.funcDeclDataMutex.RUnlock()

									if ok {
										instanceData := b.findFuncInstance(genericData, typeMap)
										if instanceData == nil {
											instanceData = b.createFuncInstance(ctx, signature, genericData, typeMap)
										}
										call.function = instanceData.linkname
										signature = instanceData.signature
									}
								}
							}
						}

						var recvArg mlir.ValueLike

						// For promoted methods (sel.Index() has > 1 element),
						// navigate through embedded fields to reach the effective
						// receiver. The last element is the method index; the
						// preceding elements are field indices.
						indices := sel.Index()
						if len(indices) > 1 {
							// Start with the address of the outer struct.
							basePtr := b.addressOf(ctx, Fun.X, location)
							currentType := sel.Recv()

							// Walk through field indices (all but last = method index).
							for _, fieldIdx := range indices[:len(indices)-1] {
								if isPointer(currentType) {
									// Load through pointer.
									ptrType := b.GetType(ctx, currentType)
									basePtr = b.emitLoad(ctx, basePtr, ptrType, location)
									currentType = currentType.(*types.Pointer).Elem()
								}

								structType := baseStructTypeOf(currentType)
								fieldType := structType.Field(fieldIdx).Type()
								fieldPtrType := b.pointerOf(ctx, fieldType)

								// GEP to the struct field.
								gepOp := goir.NewGepOperation(b.ctx,
									basePtr, b.GetType(ctx, structType),
									[]int{0, fieldIdx}, nil, []bool{false, false},
									fieldPtrType, location)
								appendOperation(ctx, gepOp)
								basePtr = resultOf(gepOp).AsValue()
								currentType = fieldType
							}

							// Now basePtr points to the field that declares the method.
							// Apply the same pointer/non-pointer logic as for direct methods.
							sigRecvType := baseType(signature.Recv().Type())
							if isPointer(currentType) {
								// The field is a pointer type (e.g., *canonMap).
								// Load it to get the actual pointer value.
								ptrType := b.GetType(ctx, currentType)
								loaded := b.emitLoad(ctx, basePtr, ptrType, location)
								if isPointer(sigRecvType) {
									recvArg = loaded
								} else {
									recvArg = b.NewTempValue(loaded).Load(ctx, location)
								}
							} else {
								if isPointer(sigRecvType) {
									recvArg = basePtr
								} else {
									loadType := b.GetType(ctx, currentType)
									recvArg = b.emitLoad(ctx, basePtr, loadType, location)
								}
							}
						} else {
							// Direct method (not promoted through embedding).
							exprType := baseType(recvT)
							sigRecvType := baseType(signature.Recv().Type())
							if isPointer(exprType) {
								// The expression yields *T
								if isPointer(sigRecvType) {
									// signature wants *T: use directly
									recvArg = b.emitExpr(ctx, Fun.X)[0]
								} else {
									// signature wants T: load
									ptr := b.emitExpr(ctx, Fun.X)[0]
									recvArg = b.NewTempValue(ptr).Load(ctx, location)
								}
							} else {
								// The expression yields T
								addr := b.addressOf(ctx, Fun.X, location)
								if isPointer(sigRecvType) {
									// signature wants *T: pass address
									recvArg = addr
								} else {
									// signature wants T: load
									recvArg = b.NewTempValue(addr).Load(ctx, location)
								}
							}
						}

						// Append the receiver value to the argument list.
						call.args = append(call.args, recvArg)
					}

				default:
					panic("unhandled")
				}

			} else {
				// The selection is actually a qualified identifier.
				funcObj := funcObj.(*types.Func)
				call.calleeType = calleeIsSymbol
				call.function = qualifiedFuncName(funcObj)
			}
		case *ast.IndexExpr:
			// Distinguish generic instantiation (e.g. genericFunc[T]()) from
			// regular index expressions (e.g. callbacks[i]()).
			isGeneric := false
			switch X := Fun.X.(type) {
			case *ast.Ident:
				if obj, ok := info.Uses[X]; ok {
					_, isGeneric = obj.Type().(*types.Signature)
				}
			case *ast.SelectorExpr:
				if obj, ok := info.Uses[X.Sel]; ok {
					_, isGeneric = obj.Type().(*types.Signature)
				}
			}

			if isGeneric {
				// Generic type parameter instantiation — unwrap.
				call.typeMap = resolveTypeParams(ctx, expr, info)
				ctx = newContextWithTypeMap(ctx, call.typeMap)
				calleeExpr = Fun.X
				continue
			}
			// Regular index expression (array/slice/map) returning a callable.
			call.calleeType = calleeIsClosure
			call.callee = b.emitExpr(ctx, Fun)[0]
			// Derive the element (callable) type from the collection type.
			elemType := funcObj.Type()
			switch t := baseType(elemType).(type) {
			case *types.Array:
				elemType = t.Elem()
			case *types.Slice:
				elemType = t.Elem()
			case *types.Map:
				elemType = t.Elem()
			}
			signature = baseType(elemType).(*types.Signature)
		case *ast.IndexListExpr:
			// Resolve type parameters.
			call.typeMap = resolveTypeParams(ctx, expr, info)
			calleeExpr = Fun.X
			ctx = newContextWithTypeMap(ctx, call.typeMap)
			continue
		default:
			panic("unhandled")
		}

		if signature == nil {
			panic("signature is nil")
		}

		// Is the callee a generic function?
		if !instanceResolved && (signature.TypeParams().Len() > 0 || signature.RecvTypeParams().Len() > 0) {
			// Need to instantiate this generic function.
			b.funcDeclDataMutex.RLock()
			data, ok := b.genericFuncs[call.function]
			b.funcDeclDataMutex.RUnlock()
			if !ok {
				b.funcDeclDataMutex.RLock()
				decl := b.ungeneratedFuncs[call.function]
				b.funcDeclDataMutex.RUnlock()

				if decl != nil {
					data = b.addFunctionDecl(ctx, decl)
				}
			}

			if data != nil {
				typeMap := resolveTypeParams(ctx, expr, info)
				targs := make([]types.Type, len(typeMap))
				for index, typ := range typeMap {
					targs[index] = typ
				}

				ictx := types.NewContext()
				newType, err := types.Instantiate(ictx, signature, targs, false)
				if err != nil {
					panic(err.Error())
				}

				signature = newType.(*types.Signature)

				instanceData := b.createFuncInstance(ctx, signature, data, typeMap)
				call.function = instanceData.linkname
				signature = instanceData.signature
			}
		}

		// Evaluate all arguments to the call.
		callArgs := b.emitCallArgs(ctx, signature, expr)
		if len(callArgs) != signature.Params().Len() {
			panic("len(callArgs) != signature.Params().Len()")
		}

		call.args = append(call.args, callArgs...)

		// Collect result types.
		call.results = make([]mlir.TypeLike, 0, signature.Results().Len())
		for result := range signature.Results().Variables() {
			call.results = append(call.results, b.GetStoredType(ctx, result.Type()))
		}

		call.signature = signature

		return call
	}
}

func (b *Builder) emitCallExpr(ctx context.Context, expr *ast.CallExpr) []mlir.ValueLike {
	location := b.location(ctx, expr.Lparen)
	info := currentInfo(ctx)
	tv := info.Types[expr.Fun]

	if tv.IsBuiltin() {
		// Emit the respective runtime call.
		return b.emitBuiltinCall(ctx, expr)
	} else if b.isIntrinsic(ctx, expr) {
		return b.emitIntrinsic(ctx, expr)
	} else if tv.IsType() {
		srcType := b.typeOf(ctx, expr.Args[0])
		destType := tv.Type

		if isNil(srcType) {
			// Emit the zero value of the destination type.
			return []mlir.ValueLike{
				b.emitZeroValue(ctx, destType, location),
			}
		}

		// Evaluate the value to convert.
		X := b.emitExpr(ctx, expr.Args[0])[0]

		// Perform type conversion.
		value := b.emitTypeConversion(ctx, X, srcType, destType, location)
		return []mlir.ValueLike{value}
	} else {
		opArgs := b.extractCallOpArgs(ctx, expr)
		switch opArgs.calleeType {
		case calleeIsClosure:
			signatureTypeAttr := mlir.NewTypeAttr(b.GetType(ctx, opArgs.signature))
			op := goir.NewClosureCallOperation(
				b.ctx, signatureTypeAttr, opArgs.callee, opArgs.results, opArgs.args, location)
			appendOperation(ctx, op)
			return resultsOf(op)
		case calleeIsInterface:
			op := goir.NewInterfaceCall(
				b.ctx, opArgs.function, opArgs.results, opArgs.callee, opArgs.args, location)
			appendOperation(ctx, op)
			return resultsOf(op)
		case calleeIsSymbol:
			// Emit the function that will be called.
			symbol := b.resolveSymbol(opArgs.function)

			b.queueJob(ctx, symbol)

			op := goir.NewCallOperation(b.ctx, symbol, opArgs.results, opArgs.args, location)
			appendOperation(ctx, op)
			return resultsOf(op)
		default:
			panic("unhandled")
		}
	}
}

func (b *Builder) createSyntheticClosureSignature(ctx context.Context, signature *types.Signature) goir.FunctionType {
	inputTypes := make([]mlir.TypeLike, signature.Params().Len()+1)
	inputTypes[0] = b.ptr
	for i := 0; i < signature.Params().Len(); i++ {
		inputTypes[i+1] = b.GetStoredType(ctx, signature.Params().At(i).Type())
	}

	resultTypes := make([]mlir.TypeLike, signature.Results().Len())
	for i := 0; i < signature.Results().Len(); i++ {
		resultTypes[i] = b.GetStoredType(ctx, signature.Results().At(i).Type())
	}

	T := goir.NewFunctionType(b.ctx, nil, inputTypes, resultTypes)
	return T
}

func (b *Builder) emitCallArgs(ctx context.Context, signature *types.Signature, expr *ast.CallExpr) []mlir.ValueLike {
	location := b.location(ctx, expr.Pos())
	argValues := make([]mlir.ValueLike, len(expr.Args))
	for i, expr := range expr.Args {
		argValues[i] = b.emitExpr(ctx, expr)[0]
	}

	// Handle interface and function-value arguments.
	argTypes := make([]types.Type, len(expr.Args))
	for i := range expr.Args {
		argT := b.typeOf(ctx, expr.Args[i])
		argTypes[i] = argT

		paramT := signature.Params().At(min(i, signature.Params().Len()-1)).Type()

		switch baseType(paramT).(type) {
		case *types.Interface:
			if !isNil(argT) && !types.Identical(paramT, argT) {
				// Resolve TypeParams to their concrete types before checking
				// whether the argument is an interface. A TypeParam's underlying
				// type is its constraint interface, but the emitted value is the
				// concrete instantiation type (e.g., a struct, not an interface).
				resolvedArgT := resolveType(ctx, argT)
				if types.IsInterface(baseType(resolvedArgT)) {
					// Convert from interface A to interface B.
					argValues[i] = b.emitChangeType(ctx, paramT, argValues[i], location)
				} else {
					// Create an interface value from the value expression.
					argValues[i] = b.emitInterfaceValue(ctx, paramT, resolvedArgT, argValues[i], location)
				}
			}
		case *types.Signature:
			// Wrap raw function pointers into the _func struct. Values that
			// are already the _func struct type (e.g., closures, variables of
			// function type) must not be wrapped again.
			if ptrT, ok := goir.AsPointerType(argValues[i].Type()); ok {
				elementT := ptrT.ElementType()
				if !elementT.IsNull() && goir.TypeIsAFunctionType(elementT) {
					argValues[i] = b.createFunctionValue(ctx, argValues[i], nil, 0, location)
				}
			}
		}
	}

	return b.emitVariadicArgs(ctx, signature, argTypes, argValues, location)
}

func (b *Builder) emitGoStatement(ctx context.Context, stmt *ast.GoStmt) {
	location := b.location(ctx, stmt.Pos())
	opArgs := b.extractCallOpArgs(ctx, stmt.Call)

	// a //sigo:stacksize N pragma on this `go` statement (or on
	// the FuncLit it launches when written `go func(){...}()`) overrides
	// the called function's own pragma. For the closure path we patch the
	// _func value's stackSize field by inserting at index 2 before passing
	// to GoOperation3.
	stmtSize := int64(b.config.Program.NodeStackSize[stmt])

	switch opArgs.calleeType {
	case calleeIsClosure:
		callee := opArgs.callee
		if stmtSize > 0 {
			stackSizeValue := b.emitConstInt(ctx, stmtSize, b.uiptr, location)
			insertOp := goir.NewInsertOperation(b.ctx, 2, stackSizeValue, callee, b._func, location)
			appendOperation(ctx, insertOp)
			callee = resultOf(insertOp).AsValue()
		}
		signatureTypeAttr := mlir.NewTypeAttr(b.GetType(ctx, opArgs.signature))
		op := goir.NewGoOperation3(b.ctx, signatureTypeAttr, callee, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsInterface:
		// Interface-method goroutines: the size hint is on the receiver's
		// concrete func value, not addressable here. GoStmt-level pragma
		// for this case is a no-op for now.
		op := goir.NewGoOperation4(b.ctx, opArgs.callee, opArgs.function, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsSymbol:
		// If the callee or this `go` statement has a //sigo:stacksize
		// pragma, route through the closure path so the size is carried
		// in the _func.stackSize field. The direct GoOperation1 / symbol
		// path has no slot for a per-launch stack size — addGoroutine
		// reads it only from f.stackSize. The cost is one shim
		// indirection on the goroutine entry; the win is no C++ change.
		symInfo := b.config.Program.Symbols.GetSymbolInfo(opArgs.function)
		if symInfo.StackSize != 0 || stmtSize > 0 {
			if stmtSize == 0 {
				stmtSize = int64(symInfo.StackSize)
			}

			callee := b.emitFuncReferenceValue(ctx, opArgs.function, opArgs.signature, location)
			// emitFuncReferenceValue already baked in SymbolInfo.StackSize.
			// If the GoStmt pragma overrides, patch field index 2.
			if stmtSize > 0 {
				stackSizeValue := b.emitConstInt(ctx, stmtSize, b.uiptr, location)
				insertOp := goir.NewInsertOperation(b.ctx, 2, stackSizeValue, callee, b._func, location)
				appendOperation(ctx, insertOp)
				callee = resultOf(insertOp).AsValue()
			}
			signatureTypeAttr := mlir.NewTypeAttr(b.GetType(ctx, opArgs.signature))
			op := goir.NewGoOperation3(b.ctx, signatureTypeAttr, callee, opArgs.args, location)
			appendOperation(ctx, op)
			break
		}

		// No size override: direct symbol launch (avoids shim indirection).
		symbol := b.resolveSymbol(opArgs.function)
		b.queueJob(ctx, symbol)

		op := goir.NewGoOperation1(b.ctx, symbol, opArgs.args, location)
		appendOperation(ctx, op)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitDeferStatement(ctx context.Context, stmt *ast.DeferStmt) {
	location := b.location(ctx, stmt.Pos())
	opArgs := b.extractCallOpArgs(ctx, stmt.Call)
	switch opArgs.calleeType {
	case calleeIsClosure:
		signatureTypeAttr := mlir.NewTypeAttr(b.GetType(ctx, opArgs.signature))
		op := goir.NewDeferOperation3(b.ctx, signatureTypeAttr, opArgs.callee, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsInterface:
		op := goir.NewDeferOperation4(b.ctx, opArgs.callee, opArgs.function, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsSymbol:
		// Emit the function that will be called.
		symbol := b.resolveSymbol(opArgs.function)
		b.queueJob(ctx, symbol)

		op := goir.NewDeferOperation1(b.ctx, symbol, opArgs.args, location)
		appendOperation(ctx, op)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitVariadicArgs(ctx context.Context, signature *types.Signature, argTypes []types.Type, args []mlir.ValueLike, location mlir.LocationLike) []mlir.ValueLike {
	if signature.Variadic() {
		variadicBegin := signature.Params().Len() - 1
		numVariadicArgs := len(args) - variadicBegin
		variadicParamType := signature.Params().At(variadicBegin).Type()

		// Handle append([]byte, string...) — the param type is string, not a slice.
		variadicArgType, isSlice := variadicParamType.(*types.Slice)
		if !isSlice {
			// Special case: string spread into []byte (append([]byte, string...)).
			// Convert the string to a byte slice so the types match at the MLIR level.
			if typeHasFlags(variadicParamType, types.IsString) && variadicBegin < len(args) {
				byteSliceType := types.NewSlice(types.Typ[types.Byte])
				args[variadicBegin] = b.emitTypeConversion(ctx, args[variadicBegin], variadicParamType, byteSliceType, location)
			}
			return args
		}

		elementType := variadicArgType.Elem()
		elementT := b.GetStoredType(ctx, elementType)
		elementPtrT := b.GetStoredType(ctx, types.NewPointer(elementType))

		if numVariadicArgs == 0 {
			// No variadic arguments provided — pass a nil slice.
			varArg := b.emitZeroValue(ctx, variadicArgType, location)
			args = append(args[:variadicBegin], varArg)
			return args
		}

		if args[variadicBegin].Type().Equal(b.GetStoredType(ctx, variadicArgType)) {
			// This is ellipsis (...).
			return args
		}

		// Create the backing array for the slice that will contain the variadic arguments.
		allocaOp := goir.NewAllocaOperation(b.ctx, b.ptr, elementT, numVariadicArgs, false, location)
		appendOperation(ctx, allocaOp)

		// Fill the backing array.
		for i, arg := range args[variadicBegin:] {
			argT := argTypes[variadicBegin+i]

			// Gep into the backing array to the position where the current argument should be stored.
			gepOp := goir.NewGepOperation(
				b.ctx, resultOf(allocaOp), elementT, []int{i}, nil, []bool{false}, elementPtrT, location)
			appendOperation(ctx, gepOp)

			// Handle interface type conversion.
			switch baseType(elementType).(type) {
			case *types.Interface:
				if !isNil(argT) && !types.Identical(elementType, argT) {
					resolvedArgT := resolveType(ctx, argT)
					if types.IsInterface(baseType(resolvedArgT)) {
						// Convert from interface A to interface B.
						arg = b.emitChangeType(ctx, elementType, arg, location)
					} else {
						// Create an interface value from the value expression.
						arg = b.emitInterfaceValue(ctx, elementType, resolvedArgT, arg, location)
					}
				}
			}

			// Store the argument value.
			b.emitStore(ctx, arg, resultOf(gepOp), location)
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

func (b *Builder) createInterfaceCallWrapper2(ctx context.Context, symbol string, callee string, iface *types.Interface, signature *types.Signature, argTypes []types.Type) thunkType {
	b.thunkMutex.Lock()
	defer b.thunkMutex.Unlock()

	// Look up the thunk in the symbol table first.
	if _, ok := b.thunks[symbol]; !ok {
		// Prepend the interface type to the beginning of the argument pack type list.
		argTypes = append([]types.Type{iface}, argTypes...)
		vars := make([]*types.Var, len(argTypes))
		for i := range argTypes {
			vars[i] = types.NewVar(token.NoPos, nil, fmt.Sprintf("arg$%d", i), argTypes[i])
		}

		// Create the argument struct type.
		argPackT := types.NewStruct(vars, nil)
		argPackPtrT := types.NewPointer(argPackT)
		argPackPtrType := b.GetType(ctx, argPackPtrT)

		// Any argument excluded from the argument pack MUST be passed to the resulting thunk directly.
		// NOTE: The interface value is added to the parameter count.
		paramTypes := []mlir.TypeLike{argPackPtrType}
		paramVars := []*types.Var{types.NewVar(token.NoPos, nil, "param$0", argPackPtrT)}
		for i := len(argTypes); i < signature.Params().Len()+1; i++ {
			paramT := signature.Params().At(i).Type()
			paramTypes = append(paramTypes, b.GetStoredType(ctx, paramT))
			paramVars = append(paramVars, types.NewVar(token.NoPos, nil, fmt.Sprintf("param$%d", i+1), paramT))
		}
		paramLocs := make([]mlir.LocationLike, len(paramTypes))
		fill(paramLocs, b._noLoc)

		// Collect the result types.
		resultTypes := make([]mlir.TypeLike, 0, signature.Results().Len())
		resultVars := make([]*types.Var, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			resultT := signature.Results().At(i).Type()
			resultTypes = append(resultTypes, b.GetStoredType(ctx, resultT))
			resultVars = append(resultVars, types.NewVar(token.NoPos, nil, fmt.Sprintf("result$%d", i), resultT))
		}

		syntheticSig := types.NewSignatureType(nil, nil, nil,
			types.NewTuple(paramVars...), types.NewTuple(resultVars...), false)

		// Create thunk to wrap the method call.
		region := mlir.NewRegion()
		ctx = newContextWithRegion(ctx, region)

		entryBlock := mlir.NewBlock(paramTypes, paramLocs)
		region.AppendOwnedBlock(entryBlock)
		buildBlock(ctx, entryBlock, func() {
			argPackPtrValue := entryBlock.Argument(0)
			args := b.unpackArgPack(ctx, argTypes, argPackPtrValue, b._noLoc)

			// Gather the remaining arguments
			for i := 1; i < entryBlock.NumArguments(); i++ {
				args = append(args, entryBlock.Argument(i))
			}

			// Call the method.
			callOp := goir.NewInterfaceCall(b.ctx, callee, resultTypes, args[0], args[1:], b._noLoc)
			appendOperation(ctx, callOp)

			// Return the results.
			returnOp := goir.NewReturnOperation(b.ctx, resultsOf(callOp), b._noLoc)
			appendOperation(ctx, returnOp)
		})

		// Create the function operation for this thunk.
		thunkFuncType := goir.NewFunctionType(b.ctx, nil, paramTypes, resultTypes)
		funcOp := mlir.NewOperationState("go.func", b._noLoc).
			AddOwnedRegions(region).
			AddAttributes(
				mlir.NewNamedAttribute("function_type", mlir.NewTypeAttr(thunkFuncType)),
				mlir.NewNamedAttribute("sym_name", mlir.NewStringAttr(b.config.Ctx, symbol)),
				mlir.NewNamedAttribute("sym_visibility", mlir.NewStringAttr(b.config.Ctx, "private"))).
			Create()

		// This operation will be added later safely.
		b.addToModuleMutex.Lock()
		b.addToModule[symbol] = funcOp
		b.addToModuleMutex.Unlock()

		b.thunks[symbol] = struct{}{}

		result := thunkType{
			t: thunkFuncType,
			s: syntheticSig,
		}
		b.thunkTypes[symbol] = result
		return result
	}

	return b.thunkTypes[symbol]
}

func (b *Builder) emitCallArgs2(ctx context.Context, args []ast.Expr) []mlir.ValueLike {
	values := make([]mlir.ValueLike, len(args))
	for i, expr := range args {
		values[i] = b.emitExpr(ctx, expr)[0]
	}
	return values
}

// resolveTypeInContext recursively walks a type and replaces any TypeParams
// using the provided mapping. This is needed when a type arg from a receiver
// (e.g., *indirect[K, V]) contains TypeParams from an enclosing generic scope.
func resolveTypeInContext(T types.Type, mapping TypeParamMap) types.Type {
	if mapping == nil {
		return T
	}
	switch T := T.(type) {
	case *types.TypeParam:
		if resolved := mapping[T.Index()]; resolved != nil {
			return resolved
		}
		return T
	case *types.Pointer:
		elem := resolveTypeInContext(T.Elem(), mapping)
		if elem == T.Elem() {
			return T
		}
		return types.NewPointer(elem)
	case *types.Slice:
		elem := resolveTypeInContext(T.Elem(), mapping)
		if elem == T.Elem() {
			return T
		}
		return types.NewSlice(elem)
	case *types.Array:
		elem := resolveTypeInContext(T.Elem(), mapping)
		if elem == T.Elem() {
			return T
		}
		return types.NewArray(elem, T.Len())
	case *types.Map:
		key := resolveTypeInContext(T.Key(), mapping)
		val := resolveTypeInContext(T.Elem(), mapping)
		if key == T.Key() && val == T.Elem() {
			return T
		}
		return types.NewMap(key, val)
	case *types.Chan:
		elem := resolveTypeInContext(T.Elem(), mapping)
		if elem == T.Elem() {
			return T
		}
		return types.NewChan(T.Dir(), elem)
	case *types.Named:
		typeArgs := T.TypeArgs()
		if typeArgs == nil || typeArgs.Len() == 0 {
			return T
		}
		newArgs := make([]types.Type, typeArgs.Len())
		changed := false
		for i := 0; i < typeArgs.Len(); i++ {
			newArgs[i] = resolveTypeInContext(typeArgs.At(i), mapping)
			if newArgs[i] != typeArgs.At(i) {
				changed = true
			}
		}
		if !changed {
			return T
		}
		inst, err := types.Instantiate(nil, T.Origin(), newArgs, false)
		if err != nil {
			panic(fmt.Sprintf("failed to instantiate type in resolveTypeInContext: %v", err))
		}
		return inst
	default:
		return T
	}
}

func resolveTypeParams(ctx context.Context, callExpr *ast.CallExpr, info *types.Info) TypeParamMap {
	currentMapping := currentTypeMap(ctx)

	mapping := TypeParamMap{}
	switch Fun := callExpr.Fun.(type) {
	case *ast.Ident:
		// Look up this instance from the type checker info directly.
		instance, ok := info.Instances[Fun]
		if !ok {
			return nil
		}

		signature, ok := instance.Type.(*types.Signature)
		if !ok {
			return nil
		}

		var createTypeMapping func(generic types.Type, concrete types.Type)
		createTypeMapping = func(generic types.Type, concrete types.Type) {
			switch generic := generic.(type) {
			case *types.Array:
				concrete := concrete.(*types.Array)
				createTypeMapping(generic.Elem(), concrete.Elem())
			case *types.Chan:
				concrete := concrete.(*types.Chan)
				createTypeMapping(generic.Elem(), concrete.Elem())
			case *types.Map:
				concrete := concrete.(*types.Map)
				createTypeMapping(generic.Key(), concrete.Key())
				createTypeMapping(generic.Elem(), concrete.Elem())
			case *types.Pointer:
				concrete := concrete.(*types.Pointer)
				createTypeMapping(generic.Elem(), concrete.Elem())
			case *types.Slice:
				concrete := concrete.(*types.Slice)
				createTypeMapping(generic.Elem(), concrete.Elem())
			case *types.Named:
				if concrete, ok := concrete.(*types.Named); ok {
					for i := 0; i < generic.TypeArgs().Len(); i++ {
						createTypeMapping(generic.TypeArgs().At(i), concrete.TypeArgs().At(i))
					}
				}
			case *types.TypeParam:
				// If concrete is still a TypeParam and we're inside an
				// instantiated generic, resolve through the outer mapping.
				if concreteTP, ok := concrete.(*types.TypeParam); ok && currentMapping != nil {
					if resolved := currentMapping[concreteTP.Index()]; resolved != nil {
						mapping[generic.Index()] = resolved
						break
					}
				}
				mapping[generic.Index()] = concrete
			}
		}

		if signature.Recv() != nil {
			createTypeMapping(signature.Recv().Origin().Type(), signature.Recv().Type())
		}

		for obj := range signature.Params().Variables() {
			createTypeMapping(obj.Origin().Type(), obj.Type())
		}

		for obj := range signature.Results().Variables() {
			createTypeMapping(obj.Origin().Type(), obj.Type())
		}

	case *ast.IndexExpr:
		var ident *ast.Ident
		switch X := Fun.X.(type) {
		case *ast.Ident:
			ident = X
		case *ast.SelectorExpr:
			ident = X.Sel
		default:
			panic("unhandled")
		}

		obj, ok := info.Uses[ident]
		if !ok {
			return nil
		}

		signature, ok := obj.Type().(*types.Signature)
		if !ok {
			return nil
		}

		typeParams := signature.TypeParams()
		argExpr := Fun.Index
		param := typeParams.At(0)
		argType := info.TypeOf(argExpr) // Resolve the type from the AST expression.
		if argType == nil {
			return nil
		}

		switch argType := argType.(type) {
		case *types.TypeParam:
			// Look up in the current map.
			if currentMapping == nil {
				panic("type parameter cannot be resolved")
			}
			mapping[param.Index()] = currentMapping[param.Index()]
		default:
			mapping[param.Index()] = argType
		}
	case *ast.IndexListExpr:
		var ident *ast.Ident
		switch X := Fun.X.(type) {
		case *ast.Ident:
			ident = X
		case *ast.SelectorExpr:
			ident = X.Sel
		default:
			panic("unhandled")
		}

		obj, ok := info.Uses[ident]
		if !ok {
			return nil
		}

		signature, ok := obj.Type().(*types.Signature)
		if !ok {
			return nil
		}

		typeParams := signature.TypeParams()
		typeArgExprs := Fun.Indices
		for i, argExpr := range typeArgExprs {
			param := typeParams.At(i)
			argType := info.TypeOf(argExpr) // Resolve the type from the AST expression.
			if argType == nil {
				return nil
			}

			switch argType := argType.(type) {
			case *types.TypeParam:
				// Look up in the current map.
				if currentMapping == nil {
					panic("type parameter cannot be resolved")
				}
				mapping[param.Index()] = currentMapping[param.Index()]
			default:
				mapping[param.Index()] = argType
			}
		}
	case *ast.SelectorExpr:
		// First, check if the selector itself is a generic function instance
		// (package-qualified call like unique.Make(addrDetail{})).
		if instance, ok := info.Instances[Fun.Sel]; ok {
			signature, ok := instance.Type.(*types.Signature)
			if !ok {
				return nil
			}

			var createTypeMapping func(generic types.Type, concrete types.Type)
			createTypeMapping = func(generic types.Type, concrete types.Type) {
				switch generic := generic.(type) {
				case *types.Array:
					concrete := concrete.(*types.Array)
					createTypeMapping(generic.Elem(), concrete.Elem())
				case *types.Chan:
					concrete := concrete.(*types.Chan)
					createTypeMapping(generic.Elem(), concrete.Elem())
				case *types.Map:
					concrete := concrete.(*types.Map)
					createTypeMapping(generic.Key(), concrete.Key())
					createTypeMapping(generic.Elem(), concrete.Elem())
				case *types.Pointer:
					concrete := concrete.(*types.Pointer)
					createTypeMapping(generic.Elem(), concrete.Elem())
				case *types.Slice:
					concrete := concrete.(*types.Slice)
					createTypeMapping(generic.Elem(), concrete.Elem())
				case *types.Named:
					if concrete, ok := concrete.(*types.Named); ok {
						for i := 0; i < generic.TypeArgs().Len(); i++ {
							createTypeMapping(generic.TypeArgs().At(i), concrete.TypeArgs().At(i))
						}
					}
				case *types.TypeParam:
					if concreteTP, ok := concrete.(*types.TypeParam); ok && currentMapping != nil {
						if resolved := currentMapping[concreteTP.Index()]; resolved != nil {
							mapping[generic.Index()] = resolved
							break
						}
					}
					mapping[generic.Index()] = concrete
				}
			}

			if signature.Recv() != nil {
				createTypeMapping(signature.Recv().Origin().Type(), signature.Recv().Type())
			}

			for obj := range signature.Params().Variables() {
				createTypeMapping(obj.Origin().Type(), obj.Type())
			}

			for obj := range signature.Results().Variables() {
				createTypeMapping(obj.Origin().Type(), obj.Type())
			}
			break
		}

		// Otherwise, treat as a method call on a receiver type.
		receiverType := info.TypeOf(Fun.X)
		if receiverType == nil {
			return nil
		}

		var namedType *types.Named
		if ptr, isPtr := receiverType.(*types.Pointer); isPtr {
			namedType, _ = ptr.Elem().(*types.Named)
		} else {
			namedType, _ = receiverType.(*types.Named)
		}

		if namedType == nil {
			return nil
		}

		typeArgs := namedType.TypeArgs()
		if typeArgs == nil || typeArgs.Len() == 0 {
			return nil
		}

		origin := namedType.Origin()
		if origin == nil {
			return nil
		}
		typeParams := origin.TypeParams()

		if typeParams.Len() != typeArgs.Len() {
			return nil
		}

		for i := 0; i < typeParams.Len(); i++ {
			param := typeParams.At(i)
			argType := typeArgs.At(i)
			switch argType := argType.(type) {
			case *types.TypeParam:
				// Look up in the current map.
				if currentMapping == nil {
					panic("type parameter cannot be resolved")
				}
				mapping[param.Index()] = currentMapping[argType.Index()]
			default:
				// Resolve any nested TypeParams through the current mapping.
				mapping[param.Index()] = resolveTypeInContext(argType, currentMapping)
			}
		}
	}

	return mapping
}
