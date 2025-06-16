package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/sigo/mlir"
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
	litType := b.typeOf(ctx, expr)
	arrayType := baseType(litType).(*types.Array)
	T := b.GetStoredType(ctx, litType)

	// Create the zero value of the array type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, T, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp)

	// Insert each array element.
	for i, e := range expr.Elts {
		elementT := arrayType.Elem()

		// Evaluate the array element value.
		var elementValue mlir.Value
		switch e := e.(type) {
		case *ast.KeyValueExpr:
			elementValue = b.emitExpr(ctx, e.Value)[0]
		default:
			elementValue = b.emitExpr(ctx, e)[0]
		}

		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := b.typeOf(ctx, e)
			if !isNil(valueT) && !types.Identical(elementT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, elementT, elementValue, location)
				} else {
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
	litType := b.typeOf(ctx, expr)
	mapType := baseType(litType).(*types.Map)
	mapT := b.GetStoredType(ctx, litType)

	// Emit the capacity value.
	capacityVal := b.emitConstInt(ctx, int64(len(expr.Elts)), b.si, location)

	// Create the map value.
	makeOp := mlir.GoCreateMakeMapOperation(b.ctx, mapT, capacityVal, location)
	appendOperation(ctx, makeOp)

	// Spill the map to the stack.
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
	litType := b.typeOf(ctx, expr)
	sliceType := baseType(litType).(*types.Slice)
	elementT := sliceType.Elem()
	sliceT := b.GetStoredType(ctx, litType)

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
	litType := b.typeOf(ctx, expr)
	structType := baseType(litType).(*types.Struct)
	structT := b.GetStoredType(ctx, litType)

	// Create the zero value of the struct type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, structT, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp)

	// Set the struct elements.
	for i, e := range expr.Elts {
		elementLoc := b.location(e.Pos())
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
		case *types.Signature:
			if mlir.TypeIsAFunction(mlir.ValueGetType(elementValue)) {
				// Convert the function pointer to a func value.
				elementValue = b.createFunctionValue(ctx, elementValue, nil, elementLoc)
			}
		case *types.Interface:
			// Handle interface conversion.
			valueT := b.typeOf(ctx, valueExpr)
			if !isNil(valueT) && !types.Identical(fieldT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, fieldT, elementValue, elementLoc)
				} else {
					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, fieldT, valueT, elementValue, elementLoc)
				}
			}
		}

		// Insert the value into the struct.
		insertOp := mlir.GoCreateInsertOperation(b.ctx, uint64(index), elementValue, value, structT, elementLoc)
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
	originalSignature := b.typeOf(ctx, expr).(*types.Signature)

	// Create a synthetic signature with the context pointer as the first parameter.
	var recvTypeParams, typeParams []*types.TypeParam
	var params, results []*types.Var
	if originalSignature.RecvTypeParams() != nil {
		recvTypeParams = make([]*types.TypeParam, originalSignature.RecvTypeParams().Len())
		for i := 0; i < originalSignature.RecvTypeParams().Len(); i++ {
			recvTypeParams[i] = originalSignature.RecvTypeParams().At(i)
		}
	}

	if originalSignature.TypeParams() != nil {
		typeParams = make([]*types.TypeParam, originalSignature.TypeParams().Len())
		for i := 0; i < originalSignature.TypeParams().Len(); i++ {
			typeParams[i] = originalSignature.TypeParams().At(i)
		}
	}

	params = make([]*types.Var, originalSignature.Params().Len()+1)
	params[0] = types.NewVar(token.NoPos, nil, "captures", types.Typ[types.UnsafePointer])
	for i := 0; i < originalSignature.Params().Len(); i++ {
		params[i+1] = originalSignature.Params().At(i)
	}

	results = make([]*types.Var, originalSignature.Results().Len())
	for i := 0; i < originalSignature.Results().Len(); i++ {
		results[i] = originalSignature.Results().At(i)
	}

	signature := types.NewSignatureType(nil, recvTypeParams, typeParams, types.NewTuple(params...), types.NewTuple(results...), originalSignature.Variadic())
	T := b.GetType(ctx, signature)

	// Create the function data for the anonymous function.
	anonData := &funcData{
		symbol:    fmt.Sprintf("anon_func_%s", b.locationHashString(expr.Pos())),
		linkname:  fmt.Sprintf("anon_func_%s", b.locationHashString(expr.Pos())),
		funcType:  expr.Type,
		mlirType:  T,
		signature: signature,
		body:      expr.Body,
		pos:       expr.Pos(),

		locals:         map[types.Object]Value{},
		anonymousFuncs: map[*ast.FuncLit]*funcData{},
		instances:      map[*types.Signature]*funcData{},
		typeMap:        map[int]types.Type{},
		loads:          map[mlir.Block]map[types.Object]mlir.Value{},
		scope:          scope,
		info:           info,

		isAnonymous: true,
	}

	// NOTE: A literal function defined at the global scope will NOT have any enclosing function.
	if enclosingData != nil {
		// Inherit the enclosing function's type mappings.
		anonData.typeMap = enclosingData.typeMap

		// Have the enclosing function track this anonymous function.
		enclosingData.anonymousFuncs[expr] = anonData

		// Find all free variables.
		captures := map[types.Object]*FreeVar{}

		// First track all variables declared locally in the anonymous function.
		localObj := map[types.Object]struct{}{}
		ast.Inspect(expr.Body, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.GenDecl:
				for _, spec := range node.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range vs.Names {
							if obj := b.objectOf(ctx, name); obj != nil {
								localObj[obj] = struct{}{}
							}
						}
					}
				}

			case *ast.FuncLit:
				if node == expr { // only the current func
					for _, field := range node.Type.Params.List {
						for _, name := range field.Names {
							if obj := b.objectOf(ctx, name); obj != nil {
								localObj[obj] = struct{}{}
							}
						}
					}
				}

			case *ast.AssignStmt:
				if node.Tok == token.DEFINE {
					for _, lhs := range node.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok {
							if obj := b.objectOf(ctx, ident); obj != nil {
								localObj[obj] = struct{}{}
							}
						}
					}
				}
			}
			return true
		})

		// Collect all used identifiers inside the closure ignoring ones that are declared locally.
		used := map[types.Object]bool{}
		ast.Inspect(expr.Body, func(n ast.Node) bool {
			ident, ok := n.(*ast.Ident)
			if !ok || ident.Obj == nil {
				return true
			}

			obj := info.Uses[ident]
			if obj == nil {
				return true
			}

			// Skip if declared in the current function.
			if _, ok := localObj[obj]; ok {
				return true
			}

			// Skip parameters.
			for v := range originalSignature.Params().Variables() {
				if v == obj {
					return true
				}
			}

			// Skip package names, types, etc.
			switch obj.(type) {
			case *types.PkgName, *types.Func, *types.TypeName, *types.Const:
				return true
			}

			// Don't double-count
			if used[obj] {
				return true
			}

			// This is a valid capture candidate.
			used[obj] = true
			return true
		})

		for obj := range used {
			// You can skip this whole scope walk logic — the object already knows its scope.
			varType := b.GetStoredType(ctx, obj.Type())
			ptrType := mlir.GoCreatePointerType(varType)
			allocType := mlir.GoCreatePointerType(ptrType)

			allocaOp := mlir.GoCreateAllocaOperation(b.ctx, allocType, ptrType, 1, false, b.location(obj.Pos()))
			fv := &FreeVar{
				obj: obj,
				ptr: resultOf(allocaOp),
				T:   varType,
				b:   b,
			}

			captures[obj] = fv
			anonData.freeVars = append(anonData.freeVars, fv)
			anonData.locals[obj] = fv
		}

		/*
			for scope.Parent() != nil {
				scope = scope.Parent()
				for _, name := range scope.Names() {
					capturedObj := scope.Lookup(name)

					// Skip variables declared after this anonymous function.
					if capturedObj.Pos() > expr.Pos() {
						continue
					}

					// Ignore some object types.
					switch capturedObj.(type) {
					case *types.PkgName:
						continue
					}

					varType := b.GetStoredType(ctx, capturedObj.Type())
					ptrType := mlir.GoCreatePointerType(varType)
					allocType := mlir.GoCreatePointerType(ptrType)

					// Create an allocation to hold the pointer to the variable in the outer scope.
					// NOTE: The pointee should reside on the heap.
					allocaOp := mlir.GoCreateAllocaOperation(b.ctx, allocType, ptrType, 1, false, b.location(scope.Pos()))

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
		*/
	}

	// Get and return the address of the function.
	funcPtr := b.addressOfSymbol(ctx, anonData.linkname, b.ptr, location)

	// Create a pointer to a context value.
	var contextPtr mlir.Value
	if contextValue, contextType := anonData.createContextStructValue(ctx, b, location); contextValue != nil {
		anonData.contextType = contextType
		allocaOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, contextType, 1, false, location)
		appendOperation(ctx, allocaOp)
		contextPtr = resultOf(allocaOp)

		storeOp := mlir.GoCreateStoreOperation(b.ctx, contextValue, contextPtr, location)
		appendOperation(ctx, storeOp)
	}

	// Emit the anonymous function.
	b.emitFunc(ctx, anonData)

	// Return a func value.
	return b.createFunctionValue(ctx, funcPtr, contextPtr, location)
}
