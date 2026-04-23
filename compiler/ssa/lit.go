package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitBasicLiteral(ctx context.Context, expr *ast.BasicLit) mlir.Value {
	info := currentInfo(ctx)
	TV := info.Types[expr]
	location := b.location(ctx, expr.Pos())
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
	location := b.location(ctx, expr.Pos())
	litType := b.typeOf(ctx, expr)
	arrayType := baseType(litType).(*types.Array)
	T := b.GetStoredType(ctx, litType)

	// Create the zero value of the array type.
	zeroOp := goir.NewZeroOperation(b.ctx, T, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp).AsValue()

	// Insert each array element.
	for i, e := range expr.Elts {
		elementT := arrayType.Elem()

		// Determine the actual array index and evaluate the element value.
		index := uint64(i)
		var elementValue mlir.Value
		switch e := e.(type) {
		case *ast.KeyValueExpr:
			// Keyed element: extract the key's constant value as the array index.
			info := currentInfo(ctx)
			keyVal := info.Types[e.Key].Value
			idx, _ := constant.Int64Val(keyVal)
			index = uint64(idx)
			elementValue = b.emitExpr(ctx, e.Value)[0].AsValue()
		default:
			elementValue = b.emitExpr(ctx, e)[0].AsValue()
		}

		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := resolveType(ctx, b.typeOf(ctx, e))
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
		insertOp := goir.NewInsertOperation(b.ctx, index, elementValue, value, T, location)
		appendOperation(ctx, insertOp)
		value = resultOf(insertOp).AsValue()
	}

	return value
}

func (b *Builder) emitMapLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(ctx, expr.Pos())
	litType := b.typeOf(ctx, expr)
	mapType := baseType(litType).(*types.Map)
	mapT := b.GetStoredType(ctx, litType)
	keyT := mapType.Key()
	elementT := mapType.Elem()
	elementPtrT := b.pointerOf(ctx, elementT)

	// Emit the capacity value.
	capacityVal := b.emitConstInt(ctx, int64(len(expr.Elts)), b.si, location)

	// Create the map value.
	makeOp := goir.NewMakeMapOperation(b.ctx, mapT, capacityVal, location)
	appendOperation(ctx, makeOp)

	// Spill the map to the stack.
	mapValue := resultOf(makeOp).AsValue()

	// Insert each value into the map.
	for _, expr := range expr.Elts {
		expr := expr.(*ast.KeyValueExpr)

		// Evaluate the key and element values.
		// The key must be passed as a pointer to the map.addr operation.
		keyValue := mlir.ValueLike(b.makeCopyOf(ctx, b.emitExpr(ctx, expr.Key)[0].AsValue(), keyT, location))
		elementValue := b.emitExpr(ctx, expr.Value)[0]

		// Handle interface conversions.
		switch baseType(keyT).(type) {
		case *types.Interface:
			valueT := resolveType(ctx, b.typeOf(ctx, expr.Key))
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
			valueT := resolveType(ctx, b.typeOf(ctx, expr.Value))
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
		updateOp := goir.NewMapAddrOperation(b.ctx, elementPtrT, mapValue, keyValue, location)
		appendOperation(ctx, updateOp)
		addr := resultOf(updateOp)
		b.emitStore(ctx, elementValue, addr, location)
	}

	// Finally, return the map value.
	return mapValue
}

func (b *Builder) emitSliceLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(ctx, expr.Pos())
	litType := b.typeOf(ctx, expr)
	sliceType := baseType(litType).(*types.Slice)
	elementT := sliceType.Elem()
	sliceT := b.GetStoredType(ctx, litType)

	// Create the slice value.
	lengthVal := b.emitConstInt(ctx, int64(len(expr.Elts)), b.si, location)
	makeOp := goir.NewMakeSliceOperation(b.ctx, sliceT, lengthVal, lengthVal, location)
	appendOperation(ctx, makeOp)
	sliceVal := resultOf(makeOp).AsValue()

	// Fill the slice.
	for i, expr := range expr.Elts {
		// Evaluate the array element value.
		elementValue := b.emitExpr(ctx, expr)[0]

		// Handle interface conversion.
		switch baseType(elementT).(type) {
		case *types.Interface:
			valueT := resolveType(ctx, b.typeOf(ctx, expr))
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
		addrOp := goir.NewSliceAddrOperation(b.ctx, pointerT, sliceVal, indexVal, location)
		appendOperation(ctx, addrOp)
		addr := resultOf(addrOp)

		// Store the value at the address.
		b.emitStore(ctx, elementValue, addr, location)
	}

	return sliceVal
}

func (b *Builder) emitStructLiteral(ctx context.Context, expr *ast.CompositeLit) mlir.Value {
	location := b.location(ctx, expr.Pos())
	litType := b.typeOf(ctx, expr)
	structType := baseType(litType).(*types.Struct)
	structT := b.GetStoredType(ctx, litType)

	// Create the zero value of the struct type.
	zeroOp := goir.NewZeroOperation(b.ctx, structT, location)
	appendOperation(ctx, zeroOp)
	value := resultOf(zeroOp).AsValue()

	// Set the struct elements.
	for i, e := range expr.Elts {
		elementLoc := b.location(ctx, e.Pos())
		var index int
		var valueExpr ast.Expr

		var field *types.Var

		switch e := e.(type) {
		case *ast.KeyValueExpr:
			// Get identifier that wil be used look up the specific struct field.
			identifier := e.Key.(*ast.Ident)

			// Look up the struct field.
			index, field = findStructField(identifier.Name, structType)
			valueExpr = e.Value
		default:
			// Get the struct field information by index.
			index, field = i, structType.Field(i)
			valueExpr = e
		}
		fieldT := field.Type()

		// Evaluate the array element value.
		elementValue := b.emitExpr(ctx, valueExpr)[0]

		switch baseType(fieldT).(type) {
		case *types.Signature:
			if goir.TypeIsAFunctionType(elementValue.Type()) {
				// Convert the function pointer to a func value.
				elementValue = b.createFunctionValue(ctx, elementValue, nil, elementLoc)
			}
		case *types.Interface:
			// Handle interface conversion.
			valueT := resolveType(ctx, b.typeOf(ctx, valueExpr))
			if !isNil(valueT) && !types.Identical(fieldT, valueT) {
				if types.IsInterface(baseType(valueT)) {
					// Convert from interface A to interface B.
					elementValue = b.emitChangeType(ctx, fieldT, elementValue, elementLoc)
				} else {
					// Create an interface value from the value expression.
					elementValue = b.emitInterfaceValue(ctx, fieldT, valueT, elementValue, elementLoc)
				}
			}
		case *types.Chan:
			// Channel directions differ (e.g. chan T assigned to <-chan T).
			// All directions have the same runtime representation, so bitcast.
			valueT := b.typeOf(ctx, valueExpr)
			if !types.Identical(fieldT, valueT) {
				elementValue = b.bitcastTo(ctx, elementValue, b.GetStoredType(ctx, fieldT), elementLoc)
			}
		}

		// Insert the value into the struct.
		insertOp := goir.NewInsertOperation(b.ctx, uint64(index), elementValue, value, structT, elementLoc)
		appendOperation(ctx, insertOp)
		value = resultOf(insertOp).AsValue()
	}

	return value
}

func (b *Builder) emitFuncLiteral(ctx context.Context, expr *ast.FuncLit) mlir.Value {
	location := b.location(ctx, expr.Pos())
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
	T := b.GetType(ctx, signature).(goir.FunctionType)

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
		instances:      []*funcData{},
		typeMap:        map[int]types.Type{},
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
			ptrType := b.GetStoredType(ctx, types.NewPointer(obj.Type()))
			allocType := b.GetStoredType(ctx, types.NewPointer(types.NewPointer(obj.Type())))
			allocaOp := goir.NewAllocaOperation(b.ctx, allocType, ptrType, 1, false, b.location(ctx, obj.Pos()))
			fv := &FreeVar{
				obj: obj,
				ptr: resultOf(allocaOp).AsValue(),
				T:   varType,
				GoT: obj.Type(),
				b:   b,
			}

			captures[obj] = fv
			anonData.freeVars = append(anonData.freeVars, fv)
			anonData.locals[obj] = fv
		}
	}

	// Mark captured variables' allocas as heap-allocated so they survive after the
	// enclosing function returns. The closure holds pointers to these allocas via the
	// context struct, so they must not be freed when the enclosing stack frame is popped.
	for _, fv := range anonData.freeVars {
		if enclosingValue := b.lookupValue(ctx, fv.obj); enclosingValue != nil {
			if lv, ok := enclosingValue.(*LocalValue); ok {
				if result, ok := lv.Pointer(ctx, location).AsResult(); ok {
					goir.AllocaOperationSetIsHeap(result.OwningOperation(), true)
				}
			}
		}
	}

	// Get and return the address of the function.
	funcPtr := b.addressOfSymbol(ctx, anonData.linkname, b.ptr, location)

	// Create a pointer to a context value.
	var contextPtr mlir.Value
	if contextValue, contextType, _ := anonData.createContextStructValue(ctx, b, location); contextValue != nil {
		anonData.contextType = contextType
		allocaOp := goir.NewAllocaOperation(b.ctx, b.ptr, contextType, 1, true, location)
		appendOperation(ctx, allocaOp)
		contextPtr = resultOf(allocaOp).AsValue()

		b.emitStore(ctx, contextValue, contextPtr, location)
	}

	// Emit the anonymous function.
	b.addToJobQueue(ctx, anonData)

	// Return a func value.
	return b.createFunctionValue(ctx, funcPtr, contextPtr, location)
}
