package ssa

import (
	"context"
	"go/ast"
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
					if typeParam, ok := recvT.(*types.TypeParam); ok {
						recvT = resolveType(ctx, typeParam)
						namedRecvT := recvT.(*types.Named)

						// Find the matching method of the concrete type.
						ok := false
						for method := range namedRecvT.Methods() {
							if method.Name() == funcObj.Name() {
								funcObj = method
								ok = true
							}
						}

						if !ok {
							panic("concrete method not found")
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

						var recvArg mlir.ValueLike
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
		if signature.TypeParams().Len() > 0 || signature.RecvTypeParams().Len() > 0 {
			// Need to instantiate this generic function.
			data, ok := b.genericFuncs[call.function]
			if !ok {
				b.funcDeclDataMutex.Lock()
				decl := b.ungeneratedFuncs[call.function]
				b.funcDeclDataMutex.Unlock()

				if decl != nil {
					data = b.addFunctionDecl(ctx, decl)
				}
			}

			if data != nil {
				typeMap := resolveTypeParams(ctx, expr, info)
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
		if signature.Variadic() && (i == signature.Params().Len()-1) {
			inputTypes[i+1] = goir.NewSliceType(b.GetStoredType(ctx, signature.Params().At(i).Type()))
		} else {
			inputTypes[i+1] = b.GetStoredType(ctx, signature.Params().At(i).Type())
		}
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
		switch expr := expr.(type) {
		case *ast.Ident:
			if expr.Obj != nil {
				if _, ok := expr.Obj.Decl.(*ast.FuncDecl); ok {
					// Only create a function struct value if the identifier is that of a function declaration.
					if typeIs[*types.Signature](b.typeOf(ctx, expr)) {
						argValues[i] = b.createFunctionValue(ctx, argValues[i], nil, location)
					}
				}
			}
		}
	}

	// Handle interface arguments.
	argTypes := make([]types.Type, len(expr.Args))
	for i := range expr.Args {
		argT := b.typeOf(ctx, expr.Args[i])
		argTypes[i] = argT

		paramT := signature.Params().At(min(i, signature.Params().Len()-1)).Type()

		switch baseType(paramT).(type) {
		case *types.Interface:
			if !isNil(argT) && !types.Identical(paramT, argT) {
				if types.IsInterface(baseType(argT)) {
					// Convert from interface A to interface B.
					argValues[i] = b.emitChangeType(ctx, paramT, argValues[i], location)
				} else {
					// Create an interface value from the value expression.
					argValues[i] = b.emitInterfaceValue(ctx, paramT, argT, argValues[i], location)
				}
			}
		}
	}

	return b.emitVariadicArgs(ctx, signature, argTypes, argValues, location)
}

func (b *Builder) emitGoStatement(ctx context.Context, stmt *ast.GoStmt) {
	location := b.location(ctx, stmt.Pos())
	opArgs := b.extractCallOpArgs(ctx, stmt.Call)
	switch opArgs.calleeType {
	case calleeIsClosure:
		signatureTypeAttr := mlir.NewTypeAttr(b.GetType(ctx, opArgs.signature))
		op := goir.NewGoOperation3(b.ctx, signatureTypeAttr, opArgs.callee, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsInterface:
		op := goir.NewGoOperation4(b.ctx, opArgs.callee, opArgs.function, opArgs.args, location)
		appendOperation(ctx, op)
	case calleeIsSymbol:
		// Emit the function that will be called.
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
		variadicArgType := signature.Params().At(variadicBegin).Type().(*types.Slice)
		elementType := variadicArgType.Elem()
		elementT := b.GetStoredType(ctx, elementType)

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
				b.ctx, resultOf(allocaOp), elementT, []int{i}, nil, []bool{false}, goir.NewPointerType(elementT), location)
			appendOperation(ctx, gepOp)

			// Handle interface type conversion.
			switch baseType(elementType).(type) {
			case *types.Interface:
				if !isNil(argT) && !types.Identical(elementType, argT) {
					if types.IsInterface(baseType(argT)) {
						// Convert from interface A to interface B.
						arg = b.emitChangeType(ctx, elementType, arg, location)
					} else {
						// Create an interface value from the value expression.
						arg = b.emitInterfaceValue(ctx, elementType, argT, arg, location)
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

func (b *Builder) createInterfaceCallWrapper(ctx context.Context, symbol string, callee string, iface *types.Interface, signature *types.Signature, argTypes []mlir.TypeLike) goir.FunctionType {
	b.thunkMutex.Lock()
	defer b.thunkMutex.Unlock()

	// Look up the thunk in the symbol table first.
	if _, ok := b.thunks[symbol]; !ok {
		// Prepend the interface type to the beginning of the argument pack type list.
		argTypes = append([]mlir.TypeLike{b.GetStoredType(ctx, iface)}, argTypes...)

		// Create the argument struct type.
		argPackType := goir.NewBasicStructType(b.ctx, argTypes)

		// Any argument excluded from the argument pack MUST be passed to the resulting thunk directly.
		// NOTE: The interface value is added to the parameter count.
		paramTypes := []mlir.TypeLike{goir.NewPointerType(argPackType)}
		for i := len(argTypes); i < signature.Params().Len()+1; i++ {
			paramTypes = append(paramTypes, b.GetStoredType(ctx, signature.Params().At(i).Type()))
		}
		paramLocs := make([]mlir.LocationLike, len(paramTypes))
		fill(paramLocs, b._noLoc)

		// Collect the result types.
		resultTypes := make([]mlir.TypeLike, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			resultTypes = append(resultTypes, b.GetStoredType(ctx, signature.Results().At(i).Type()))
		}

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
		b.thunkTypes[symbol] = thunkFuncType
		return thunkFuncType
	}

	return b.thunkTypes[symbol].(goir.FunctionType)
}

func (b *Builder) emitCallArgs2(ctx context.Context, args []ast.Expr) []mlir.ValueLike {
	values := make([]mlir.ValueLike, len(args))
	for i, expr := range args {
		values[i] = b.emitExpr(ctx, expr)[0]
	}
	return values
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
			case *types.TypeParam:
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
				mapping[param.Index()] = currentMapping[param.Index()]
			default:
				mapping[param.Index()] = argType
			}
		}
	}

	return mapping
}
