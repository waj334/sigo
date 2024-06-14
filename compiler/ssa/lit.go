package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitBasicLiteral(ctx context.Context, expr *ast.BasicLit) mlir.Value {
	info := currentInfo(ctx)
	TV := info.Types[expr]
	location := b.location(expr.Pos())
	T := resolveType(ctx, TV.Type)
	return b.emitConstantValue(ctx, TV.Value, T, location)
}

func (b *Builder) emitCompositeLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	switch b.typeOf(ctx, expr).Underlying().(type) {
	case *types.Array:
		return b.emitArrayLiteral(ctx, expr)
	case *types.Map:
		return b.emitMapLiteral(ctx, expr)
	case *types.Slice:
		return b.emitSliceLiteral(ctx, expr)
	case *types.Struct:
		return b.emitStructLiteral(ctx, expr)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitArrayLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(expr.Pos())
	arrayType := b.typeOf(ctx, expr).Underlying().(*types.Array)
	T := b.GetStoredType(ctx, arrayType)

	// Create the zero value of the array type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, T, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp)

	// Insert each array element.
	for i, e := range expr.Elts {
		elementT := arrayType.Elem()

		// Evaluate the array element value.
		elementValue := b.emitExpr(ctx, e)[0]

		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := b.typeOf(ctx, e)
			if !isNil(valueT) && !types.Identical(elementT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, elementT, elementValue, location)
				} else {
					// Generate methods for named types.
					if T, ok := valueT.(*types.Named); ok {
						b.queueNamedTypeJobs(ctx, T)
					}

					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, elementT, valueT, elementValue, location)
				}
			}
		}

		// Insert the element value into the array.
		insertOp := mlir.GoCreateInsertOperation(b.ctx, uint64(i), elementValue, value, T, location)
		appendOperation(ctx, insertOp)
		value = resultOf(insertOp)
	}

	return value
}

func (b *Builder) emitMapLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(expr.Pos())
	mapType := b.typeOf(ctx, expr).Underlying().(*types.Map)
	mapT := b.GetStoredType(ctx, mapType)

	// Emit the capacity value.
	capacityVal := b.emitConstInt(ctx, int64(len(expr.Elts)), b.si, location)

	// Create the map value.
	makeOp := mlir.GoCreateMakeMapOperation(b.ctx, mapT, capacityVal, location)
	appendOperation(ctx, makeOp)
	mapValue := resultOf(makeOp)

	// Insert each value into the map.
	for _, expr := range expr.Elts {
		expr := expr.(*ast.KeyValueExpr)

		// Evaluate the key and element values.
		keyValue := b.emitExpr(ctx, expr.Key)[0]
		elementValue := b.emitExpr(ctx, expr.Value)[0]

		keyT := mapType.Key()
		elementT := mapType.Elem()

		// Handle interface conversions.
		switch baseType(keyT).(type) {
		case *types.Interface:
			valueT := b.typeOf(ctx, expr.Key)
			if !isNil(valueT) && !types.Identical(elementT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					keyValue = b.emitChangeType(ctx, keyT, keyValue, location)
				} else {
					// Generate methods for named types.
					if T, ok := valueT.(*types.Named); ok {
						b.queueNamedTypeJobs(ctx, T)
					}

					// Create an interface value from the value expression.
					keyValue = b.emitInterfaceValue(ctx, keyT, valueT, keyValue, location)
				}
			}
		}

		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := b.typeOf(ctx, expr.Value)
			if !isNil(valueT) && !types.Identical(elementT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, elementT, elementValue, location)
				} else {
					// Generate methods for named types.
					if T, ok := valueT.(*types.Named); ok {
						b.queueNamedTypeJobs(ctx, T)
					}

					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, elementT, valueT, elementValue, location)
				}
			}
		}

		// Update the map.
		updateOp := mlir.GoCreateMapUpdateOperation(b.ctx, mapValue, keyValue, elementValue, location)
		appendOperation(ctx, updateOp)
	}

	// Finally, return the map value.
	return mapValue
}

func (b *Builder) emitSliceLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(expr.Pos())
	sliceType := b.typeOf(ctx, expr).Underlying().(*types.Slice)
	elementT := sliceType.Elem()
	sliceT := b.GetStoredType(ctx, sliceType)

	// Create the slice value.
	lengthVal := b.emitConstInt(ctx, int64(len(expr.Elts)), b.si, location)
	makeOp := mlir.GoCreateMakeSliceOperation(b.ctx, sliceT, lengthVal, lengthVal, location)
	appendOperation(ctx, makeOp)
	sliceVal := resultOf(makeOp)

	// Fill the slice.
	for i, expr := range expr.Elts {
		// Evaluate the array element value.
		elementValue := b.emitExpr(ctx, expr)[0]

		// Handle interface conversion.
		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := b.typeOf(ctx, expr)
			if !isNil(valueT) && !types.Identical(elementT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, elementT, elementValue, location)
				} else {
					// Generate methods for named types.
					if T, ok := valueT.(*types.Named); ok {
						b.queueNamedTypeJobs(ctx, T)
					}

					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, elementT, valueT, elementValue, location)
				}
			}
		}

		// Calculate the address to store the value to.
		pointerT := b.pointerOf(ctx, sliceType.Elem())
		indexVal := b.emitConstInt(ctx, int64(i), b.si, location)
		addrOp := mlir.GoCreateSliceAddrOperation(b.ctx, pointerT, sliceVal, indexVal, location)
		appendOperation(ctx, addrOp)
		addr := resultOf(addrOp)

		// Store the value at the address.
		storeOp := mlir.GoCreateStoreOperation(b.ctx, elementValue, addr, location)
		appendOperation(ctx, storeOp)
	}

	return sliceVal
}

func (b *Builder) emitStructLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(expr.Pos())
	structType := b.typeOf(ctx, expr).Underlying().(*types.Struct)
	structT := b.GetStoredType(ctx, b.typeOf(ctx, expr))

	// Create the zero value of the struct type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, structT, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp)

	// Set the struct elements.
	for i, e := range expr.Elts {
		var index int
		var valueExpr ast.Expr

		var field *types.Var

		switch e := e.(type) {
		case *ast.KeyValueExpr:
			// Get identifier that wil be used look up the specific struct field.
			identifier := e.Key.(*ast.Ident)

			// Look up the struct field.
			index, field = findStructField(identifier.Name, structType)

			ctx = newContextWithLhsList(ctx, []types.Type{field.Type()})
			ctx = newContextWithRhsIndex(ctx, 0)

			valueExpr = e.Value
		default:
			// Get the struct field information by index.
			index, field = i, structType.Field(i)

			ctx = newContextWithLhsList(ctx, []types.Type{field.Type()})
			ctx = newContextWithRhsIndex(ctx, 0)

			valueExpr = e
		}
		fieldT := field.Type()

		// Evaluate the array element value.
		elementValue := b.emitExpr(ctx, valueExpr)[0]

		switch baseType(fieldT).(type) {
		case *types.Interface:
			// Handle interface conversion.
			valueT := b.typeOf(ctx, valueExpr)
			if !isNil(valueT) && !types.Identical(fieldT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, fieldT, elementValue, location)
				} else {
					// Generate methods for named types.
					if T, ok := valueT.(*types.Named); ok {
						b.queueNamedTypeJobs(ctx, T)
					}

					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, fieldT, valueT, elementValue, location)
				}
			}
		}

		// Insert the value into the struct.
		insertOp := mlir.GoCreateInsertOperation(b.ctx, uint64(index), elementValue, value, structT, location)
		appendOperation(ctx, insertOp)
		value = resultOf(insertOp)
	}

	return value
}

func (b *Builder) emitFuncLiteral(ctx context.Context, expr *ast.FuncLit) mlir.Value {
	location := b.location(expr.Pos())
	enclosingData := currentFuncData(ctx)
	info := currentInfo(ctx)
	scope := info.Scopes[expr.Type]
	signature := b.typeOf(ctx, expr).(*types.Signature)

	// Create a synthetic signature with the context pointer as the first parameter.
	var recvTypeParams, typeParams []*types.TypeParam
	var params, results []*types.Var
	if signature.RecvTypeParams() != nil {
		recvTypeParams = make([]*types.TypeParam, signature.RecvTypeParams().Len())
		for i := 0; i < signature.RecvTypeParams().Len(); i++ {
			recvTypeParams[i] = signature.RecvTypeParams().At(i)
		}
	}

	if signature.TypeParams() != nil {
		typeParams = make([]*types.TypeParam, signature.TypeParams().Len())
		for i := 0; i < signature.TypeParams().Len(); i++ {
			typeParams[i] = signature.TypeParams().At(i)
		}
	}

	params = make([]*types.Var, signature.Params().Len()+1)
	params[0] = types.NewVar(token.NoPos, nil, "captures", types.Typ[types.UnsafePointer])
	for i := 0; i < signature.Params().Len(); i++ {
		params[i+1] = signature.Params().At(i)
	}

	results = make([]*types.Var, signature.Results().Len())
	for i := 0; i < signature.Results().Len(); i++ {
		results[i] = signature.Results().At(i)
	}

	signature = types.NewSignatureType(nil, recvTypeParams, typeParams, types.NewTuple(params...), types.NewTuple(results...), signature.Variadic())
	T := b.GetType(ctx, signature)

	// Create the function data for the anonymous function.
	anonData := &funcData{
		symbol:    fmt.Sprintf("anon_func_%s", b.locationHashString(expr.Pos())),
		funcType:  expr.Type,
		mlirType:  T,
		signature: signature,
		body:      expr.Body,
		pos:       expr.Pos(),

		//locals:         map[string]Value{},
		locals:         map[types.Object]Value{},
		anonymousFuncs: map[*ast.FuncLit]*funcData{},
		instances:      map[*types.Signature]*funcData{},
		typeMap:        map[int]types.Type{},
		scope:          scope,
		info:           info,

		isAnonymous: true,
	}

	// NOTE: A literal function defined at the global scope will NOT have any enclosing function.
	if enclosingData != nil {
		// Inherit the enclosing function's type mappings.
		anonData.typeMap = enclosingData.typeMap
	}

	// Find all free variables.
	captures := map[types.Object]*FreeVar{}
	for scope.Parent() != nil {
		scope = scope.Parent()
		for _, name := range scope.Names() {
			capturedObj := scope.Lookup(name)

			// Skip variables declared after this anonymous function.
			if capturedObj.Pos() > expr.Pos() {
				continue
			}

			varType := b.GetStoredType(ctx, capturedObj.Type())
			ptrType := mlir.GoCreatePointerType(varType)
			allocType := mlir.GoCreatePointerType(ptrType)

			// Create an allocation to hold the pointer to the variable in the outer scope.
			// NOTE: The pointee should reside on the heap.
			allocaOp := mlir.GoCreateAllocaOperation(b.ctx, allocType, ptrType, nil, false, b.location(scope.Pos()))

			// Create a FreeVar.
			fv := &FreeVar{
				obj: capturedObj,
				ptr: resultOf(allocaOp),
				T:   varType,
				b:   b,
			}
			captures[capturedObj] = fv
			anonData.freeVars = append(anonData.freeVars, fv)
			anonData.locals[capturedObj] = fv
		}

		if scope == enclosingData.scope {
			// Stop examining parent scopes.
			break
		}
	}

	// NOTE: A literal function defined at the global scope will NOT have any enclosing function.
	if enclosingData != nil {
		enclosingData.anonymousFuncs[expr] = anonData
	}

	// Emit the anonymous function.
	b.emitFunc(ctx, anonData)

	// Get and return the address of the function.
	op := mlir.GoCreateAddressOfOperation(b.ctx, anonData.symbol, anonData.mlirType, location)
	appendOperation(ctx, op)
	funcPtr := resultOf(op)

	// Create a pointer to a context value.
	var contextPtr mlir.Value
	if contextValue, contextType := anonData.createContextStructValue(ctx, b, location); contextValue != nil {
		allocaOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, contextType, nil, false, location)
		appendOperation(ctx, allocaOp)
		contextPtr = resultOf(allocaOp)

		storeOp := mlir.GoCreateStoreOperation(b.ctx, contextValue, contextPtr, location)
		appendOperation(ctx, storeOp)
	}

	// Return a func value.
	return b.createFunctionValue(ctx, funcPtr, contextPtr, location)
}
