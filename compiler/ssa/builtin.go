package ssa

import (
	"context"
	"go/ast"
	"go/types"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitBuiltinCall(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	signature := b.typeOf(ctx, expr.Fun).(*types.Signature)

	// Determine the built-in function name.
	var name string
	switch T := expr.Fun.(type) {
	case *ast.Ident:
		name = T.Name
	case *ast.SelectorExpr:
		X := T.X.(*ast.Ident)
		name = X.Name + "." + T.Sel.Name
	default:
		panic("unhandled")
	}

	var results []mlir.Type

	offset := 0
	if name == "new" || name == "make" {
		// First argument is a type.
		results = append(results, b.GetStoredType(ctx, b.typeOf(ctx, expr)))
		offset = 1
	} else {
		resultType := b.typeOf(ctx, expr)
		if resultType != nil {
			switch resultType := resultType.(type) {
			case *types.Tuple:
				// Do nothing.
			default:
				results = []mlir.Type{b.GetStoredType(ctx, resultType)}
			}
		}
	}

	// Emit argument values.
	// TODO: The logic below could probably be simplified.
	var operands []mlir.Value
	if signature.Variadic() {
		operands = b.emitCallArgs(ctx, signature, expr)
	} else {
		operands = b.exprValues(ctx, expr.Args[offset:]...)
		// Handle argument type conversions.
		argIndex := 0
		for i := offset; i < signature.Params().Len(); i++ {
			paramType := signature.Params().At(i).Type()
			argExpr := expr.Args[i]
			argType := b.typeOf(ctx, argExpr)
			if !types.Identical(argType, paramType) {
				operands[argIndex] = b.emitTypeConversion(ctx, operands[argIndex], argType, paramType, location)
			}
			argIndex++
		}
	}

	// Finally, emit the built-in call.
	op := mlir.GoCreateBuiltInCallOperation(b.ctx, name, results, operands, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

/*
func (b *Builder) emitBuiltinCall(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	var name string
	switch T := expr.Fun.(type) {
	case *ast.Ident:
		name = T.Name
	case *ast.SelectorExpr:
		X := T.X.(*ast.Ident)
		name = X.Name + "." + T.Sel.Name
	default:
		panic("unhandled")
	}

	// TODO: Need to set up type inference to take in account the builtin's synthetic signature.
	var inferredTypes []types.Type
	switch name {
	case "unsafe.Sizeof":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0])}
	case "unsafe.Offsetof":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0])}
	case "unsafe.Alignof":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0])}
	case "unsafe.Add":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0]), types.Typ[types.Int]}
	case "unsafe.Slice":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0]), types.Typ[types.Int]}
	case "unsafe.SliceData":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0])}
	case "unsafe.String":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0]), types.Typ[types.Int]}
	case "unsafe.StringData":
		inferredTypes = []types.Type{types.Typ[types.String]}
	case "make":
		inferredTypes = []types.Type{b.typeOf(ctx, expr.Args[0]), types.Typ[types.Int], types.Typ[types.Int]}
	case "panic":
		inferredTypes = []types.Type{b.anyType}
	}
	ctx = newContextWithLhsList(ctx, inferredTypes)

	switch name {
	// Go Spec Builtins:
	case "append":
		return b.emitAppend(ctx, expr)
	case "cap":
		return b.emitCap(ctx, expr)
	case "clear":
		return b.emitClear(ctx, expr)
	case "close":
		b.emitClose(ctx, expr)
	case "complex":
		return b.emitComplex(ctx, expr)
	case "copy":
		return b.emitCopy(ctx, expr)
	case "delete":
		b.emitDelete(ctx, expr)
	case "imag":
		return b.emitImag(ctx, expr)
	case "len":
		return b.emitLen(ctx, expr)
	case "make":
		return b.emitMake(ctx, expr)
	case "max":
		panic("unimplemented")
	case "min":
		panic("unimplemented")
	case "new":
		return b.emitNew(ctx, expr)
	case "panic":
		b.emitPanic(ctx, expr)
	case "print":
		fallthrough
	case "println":
		b.emitPrint(ctx, expr)
	case "real":
		return b.emitReal(ctx, expr)
	case "recover":
		return b.emitRecover(ctx, expr)

	// Unsafe builtins:
	case "unsafe.Add":
		return b.emitUnsafeAdd(ctx, expr)
	case "unsafe.Alignof":
		return b.emitUnsafeAlignOf(ctx, expr)
	case "unsafe.Sizeof":
		return b.emitUnsafeSizeof(ctx, expr)
	case "unsafe.SliceData":
		return b.emitUnsafeSliceData(ctx, expr)
	case "unsafe.Slice":
		return b.emitUnsafeSlice(ctx, expr)
	default:
		panic("unknown built-in function " + name)
	}
	return nil
}

func (b *Builder) emitAppend(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.typeOf(ctx, expr.Fun)
	signature := T.(*types.Signature)
	args := b.emitCallArgs(ctx, signature, expr)
	args = append(args, b.typeInfoOf(ctx, b.typeOf(ctx, expr.Args[0]), location))
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceAppend", b.exprTypes(ctx, expr), args, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitCap(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	T := b.typeOf(ctx, expr.Args[0])
	switch {
	case typeIs[*types.Array](T):
		T := T.(*types.Array)
		return []mlir.Value{b.emitConstInt(ctx, T.Len(), b.si, location)}
	case typeIs[*types.Slice](T):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceCap", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case typeIs[*types.Chan](T):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelCap", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		panic("cannot get length of incompatible type")
	}
}

func (b *Builder) emitClear(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	value := b.valueOf(ctx, expr.Args[0])
	T := b.typeOf(ctx, expr.Args[0])
	switch T := T.(type) {
	case *types.Map:
		args := b.values(value.Pointer(ctx, location))
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.mapClear", nil, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case *types.Slice:
		args := b.values(value.Pointer(ctx, location), b.typeInfoOf(ctx, T.Elem(), location))
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceClear", nil, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		panic("cannot get length of incompatible type")
	}
}

func (b *Builder) emitClose(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelClose", nil, args, location)
	appendOperation(ctx, op)
}

func (b *Builder) emitComplex(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr))
	op := mlir.GoCreateComplexOperation(b.ctx, T, args[0], args[1], location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitCopy(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	var info mlir.Value
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	if typeHasFlags(b.typeOf(ctx, expr.Args[1]), types.IsString) {
		// Convert the string value into a slice value
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringToSlice", []mlir.Type{b._slice}, []mlir.Value{args[1]}, location)
		appendOperation(ctx, op)
		args[1] = resultOf(op)
		info = b.typeInfoOf(ctx, types.Typ[types.Byte], location)
	} else {
		elementType := b.typeOf(ctx, expr.Args[1]).Underlying().(*types.Slice).Elem()
		info = b.typeInfoOf(ctx, elementType, location)
	}
	args = append(args, info)
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceCopy", []mlir.Type{b.si}, args, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitDelete(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.mapDelete", nil, args, location)
	appendOperation(ctx, op)
}

func (b *Builder) emitImag(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	arg := b.emitExpr(ctx, expr.Args[0])[0]
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr))
	op := mlir.GoCreateImagOperation(b.ctx, T, arg, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitLen(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	T := b.typeOf(ctx, expr.Args[0])
	switch {
	case typeHasFlags(T, types.IsString):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringLen", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case typeIs[*types.Array](T):
		T := T.(*types.Array)
		return []mlir.Value{b.emitConstInt(ctx, T.Len(), b.si, location)}
	case typeIs[*types.Slice](T):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceLen", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case typeIs[*types.Map](T):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.mapLen", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case typeIs[*types.Chan](T):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelLen", []mlir.Type{b.si}, args, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		panic("cannot get length of unknown type")
	}
}

func (b *Builder) emitMake(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	switch T := b.typeOf(ctx, expr).(type) {
	case *types.Chan:
		// Is the optional capacity value present?
		var capacity mlir.Value
		if len(expr.Args) == 3 {
			capacity = b.emitExpr(ctx, expr.Args[2])[0]
			capacityType := b.typeOf(ctx, expr.Args[2])
			capacity = b.emitTypeConversion(ctx, capacity, capacityType, types.Typ[types.Int], location)
		} else {
			capacity = b.emitConstInt(ctx, 0, b.si, location)
		}

		chanT := b.GetType(ctx, T)
		makeOp := mlir.GoCreateMakeChanOperation(b.ctx, chanT, capacity, location)
		appendOperation(ctx, makeOp)
		return resultsOf(makeOp)
	case *types.Map:
		var capacity mlir.Value
		if len(expr.Args) == 1 {
			// Pass zero as the initial size value.
			capacity = b.emitConstInt(ctx, 0, b.si, location)
		} else {
			capacity = b.emitExpr(ctx, expr.Args[1])[0]
			capacityType := b.typeOf(ctx, expr.Args[1])
			capacity = b.emitTypeConversion(ctx, capacity, capacityType, types.Typ[types.Int], location)
		}

		mapT := b.GetType(ctx, T)
		makeOp := mlir.GoCreateMakeMapOperation(b.ctx, mapT, capacity, location)
		appendOperation(ctx, makeOp)
		return resultsOf(makeOp)
	case *types.Slice:
		// Evaluate the length value.
		length := b.emitExpr(ctx, expr.Args[1])[0]
		lengthType := b.typeOf(ctx, expr.Args[1])
		length = b.emitTypeConversion(ctx, length, lengthType, types.Typ[types.Int], location)

		// Is the optional capacity value present?
		var capacity mlir.Value
		if len(expr.Args) == 3 {
			capacity = b.emitExpr(ctx, expr.Args[2])[0]
			capacityType := b.typeOf(ctx, expr.Args[2])
			capacity = b.emitTypeConversion(ctx, capacity, capacityType, types.Typ[types.Int], location)
		}

		sliceT := b.GetType(ctx, T)
		makeOp := mlir.GoCreateMakeSliceOperation(b.ctx, sliceT, length, capacity, location)
		appendOperation(ctx, makeOp)
		return resultsOf(makeOp)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitNew(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr.Args[0]))
	allocaOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(T), T, 1, true, location)
	appendOperation(ctx, allocaOp)
	return resultsOf(allocaOp)
}

func (b *Builder) emitPanic(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())
	ident := expr.Fun.(*ast.Ident)
	signature := b.syntheticSignatures[ident.Name]
	args := b.emitCallArgs(ctx, signature, expr)

	// Create the successor block to continue execution from upon recover.
	successorBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, successorBlock)

	// Create the panic operation.
	op := mlir.GoCreatePanicOperation(b.ctx, args[0], successorBlock, location)
	appendOperation(ctx, op)

	// Continue control flow in the successor block.
	setCurrentBlock(ctx, successorBlock)
}

func (b *Builder) emitPrint(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())
	ident := expr.Fun.(*ast.Ident)
	signature := b.syntheticSignatures[ident.Name]
	args := b.emitCallArgs(ctx, signature, expr)

	// Create the runtime call to perform the print.
	callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime._"+ident.Name, nil, args, location)
	appendOperation(ctx, callOp)
}

func (b *Builder) emitReal(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	arg := b.emitExpr(ctx, expr.Args[0])[0]
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr))
	op := mlir.GoCreateRealOperation(b.ctx, T, arg, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitRecover(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	op := mlir.GoCreateRecoverOperation(b.ctx, b._any, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitUnsafeAdd(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)

	// Convert the pointer to an integer value.
	op := mlir.GoCreatePtrToIntOperation(b.ctx, args[0], b.uiptr, location)
	appendOperation(ctx, op)
	ptrValue := resultOf(op)

	// Convert the len value to uintptr.
	lenValue := b.emitTypeConversion(ctx, args[1], b.typeOf(ctx, expr.Args[1]), types.Typ[types.Uintptr], location)

	// Add the values.
	op = mlir.GoCreateAddIOperation(b.ctx, b.uiptr, ptrValue, lenValue, location)
	appendOperation(ctx, op)
	resultValue := resultOf(op)

	// Convert the integer back into a pointer value and then return the result.
	op = mlir.GoCreateIntToPtrOperation(b.ctx, resultValue, b.ptr, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitUnsafeAlignOf(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr.Args[0]))
	sz := mlir.GoGetTypePreferredAlignmentInBytes(T, b.config.Module)
	return []mlir.Value{b.emitConstInt(ctx, int64(sz), b.uiptr, location)}
}

func (b *Builder) emitUnsafeSizeof(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr.Args[0]))
	sz := mlir.GoGetTypeSizeInBytes(T, b.config.Module)
	return []mlir.Value{b.emitConstInt(ctx, int64(sz), b.uiptr, location)}
}

func (b *Builder) emitUnsafeSliceData(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	resultT := b.GetStoredType(ctx, b.typeOf(ctx, expr))

	// Create the runtime call to extract the slice's underlying array.
	callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceData", []mlir.Type{b.ptr}, []mlir.Value{args[0]}, location)
	appendOperation(ctx, callOp)
	resultValue := resultOf(callOp)
	resultValue = b.bitcastTo(ctx, resultValue, resultT, location)

	// Bitcast to the resulting pointer type.
	return []mlir.Value{resultValue}
}

func (b *Builder) emitUnsafeSlice(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	resultT := b.GetStoredType(ctx, b.typeOf(ctx, expr))

	// Bitcast the array pointer to unsafe.Pointer.
	ptrValue := b.bitcastTo(ctx, args[0], b.ptr, location)

	// Convert the length value to int.
	lenType := b.typeOf(ctx, expr.Args[0])
	lenValue := b.emitTypeConversion(ctx, args[1], lenType, types.Typ[types.Int], location)

	// Create the runtime call to construct the slice value.
	callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.slice", []mlir.Type{resultT}, []mlir.Value{ptrValue, lenValue}, location)
	appendOperation(ctx, callOp)
	resultValue := resultOf(callOp)
	resultValue = b.bitcastTo(ctx, resultValue, resultT, location)

	// Bitcast to the resulting pointer type.
	return []mlir.Value{resultValue}
}
*/
