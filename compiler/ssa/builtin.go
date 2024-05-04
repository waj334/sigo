package ssa

import (
	"context"
	"go/ast"
	"go/types"

	"omibyte.io/sigo/mlir"
)

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
	case "unsafe.Sizeof":
		return b.emitUnsafeSizeof(ctx, expr)
	default:
		panic("unknown built-in function " + name)
	}
	return nil
}

func (b *Builder) emitAppend(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.typeOf(expr.Fun)
	signature := T.(*types.Signature)
	args := b.exprValues(ctx, expr.Args...)
	args = b.emitVariadicArgs(ctx, signature, args, location)
	args = append(args, b.typeInfoOf(ctx, b.typeOf(expr.Args[0]), location))
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceAppend", b.exprTypes(ctx, expr), args, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitCap(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	T := b.typeOf(expr.Args[0])
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
	T := b.typeOf(expr.Args[0])
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
	T := b.GetStoredType(ctx, b.typeOf(expr))
	op := mlir.GoCreateComplexOperation(b.ctx, T, args[0], args[1], location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitCopy(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	var info mlir.Value
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	if typeHasFlags(b.typeOf(expr.Args[1]), types.IsString) {
		// Convert the string value into a slice value
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringToSlice", []mlir.Type{b._slice}, []mlir.Value{args[1]}, location)
		appendOperation(ctx, op)
		args[1] = resultOf(op)
		info = b.typeInfoOf(ctx, types.Typ[types.Byte], location)
	} else {
		elementType := b.typeOf(expr.Args[1]).Underlying().(*types.Slice).Elem()
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
	T := b.GetStoredType(ctx, b.typeOf(expr))
	op := mlir.GoCreateImagOperation(b.ctx, T, arg, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitLen(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	args := b.exprValues(ctx, expr.Args...)
	T := b.typeOf(expr.Args[0])
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
	switch T := b.typeOf(expr).(type) {
	case *types.Chan:
		// Is the optional capacity value present?
		var capacity mlir.Value
		if len(expr.Args) == 2 {
			capacity = b.emitExpr(ctx, expr.Args[2])[0]
		} else {
			capacity = b.emitConstInt(ctx, 0, b.si, location)
		}

		chanT := b.GetType(ctx, T)
		makeOp := mlir.GoCreateMakeMapOperation(b.ctx, chanT, capacity, location)
		appendOperation(ctx, makeOp)
		return resultsOf(makeOp)
	case *types.Map:
		var capacity mlir.Value
		if len(expr.Args) == 1 {
			// Pass zero as the initial size value.
			capacity = b.emitConstInt(ctx, 0, b.si, location)
		}

		mapT := b.GetType(ctx, T)
		makeOp := mlir.GoCreateMakeMapOperation(b.ctx, mapT, capacity, location)
		appendOperation(ctx, makeOp)
		return resultsOf(makeOp)
	case *types.Slice:
		// Evaluate the length value.
		length := b.emitExpr(ctx, expr.Args[1])[0]

		// Is the optional capacity value present?
		var capacity mlir.Value
		if len(expr.Args) == 3 {
			capacity = b.emitExpr(ctx, expr.Args[1])[0]
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
	T := b.GetStoredType(ctx, b.typeOf(expr.Args[0]))
	allocaOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(T), T, nil, true, location)
	appendOperation(ctx, allocaOp)
	return resultsOf(allocaOp)
}

func (b *Builder) emitPanic(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())

	// Evaluate the input argument.
	arg := b.emitExpr(ctx, expr.Args[0])[0]

	// Generate methods for named types.
	if T := b.typeOf(expr.Args[0]); T != nil {
		if T, ok := T.(*types.Named); ok {
			b.queueNamedTypeJobs(ctx, T)
		}
	}

	// Convert the argument to an interface value.
	arg = b.emitInterfaceValue(ctx, b.anyType, arg, location)

	// Create the successor block to continue execution from upon recover.
	successorBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, successorBlock)

	// Create the panic operation.
	op := mlir.GoCreatePanicOperation(b.ctx, arg, successorBlock, location)
	appendOperation(ctx, op)

	// Continue control flow in the successor block.
	setCurrentBlock(ctx, successorBlock)
}

func (b *Builder) emitPrint(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(expr.Pos())
	ident := expr.Fun.(*ast.Ident)

	// Create interface values from the variadic inputs.
	values := make([]mlir.Value, len(expr.Args))
	for i := range expr.Args {
		// Evaluate the input argument.
		arg := b.emitExpr(ctx, expr.Args[i])[0]

		// Generate methods for named types.
		if T := b.typeOf(expr.Args[i]); T != nil {
			if T, ok := T.(*types.Named); ok {
				b.queueNamedTypeJobs(ctx, T)
			}
		}

		// Create interface values from this input argument.
		values[i] = b.emitInterfaceValue(ctx, b.anyType, arg, location)
	}

	// Create the slice holding the variadic input interface values.
	constLen := b.emitConstInt(ctx, int64(len(expr.Args)), b.si, location)
	inputArrOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, b._any, constLen, false, location)
	appendOperation(ctx, inputArrOp)
	mlir.OperationMoveBefore(mlir.ValueGetDefiningOperation(constLen), inputArrOp)

	// Insert each value into the data array.
	for i, v := range values {
		gepOp := mlir.GoCreateGepOperation2(b.ctx, resultOf(inputArrOp), b._any, []any{i}, mlir.GoCreatePointerType(b._any), location)
		appendOperation(ctx, gepOp)
		storeOp := mlir.GoCreateStoreOperation(b.ctx, v, resultOf(gepOp), location)
		appendOperation(ctx, storeOp)
	}

	// Emit the arguments slice.
	inputSlice := b.emitConstSlice(ctx, resultOf(inputArrOp), len(expr.Args), location)
	inputSlice = b.bitcastTo(ctx, inputSlice, mlir.GoCreateSliceType(b._any), location)

	// Create the runtime call to perform the print.
	callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime._"+ident.Name, nil, []mlir.Value{inputSlice}, location)
	appendOperation(ctx, callOp)
}

func (b *Builder) emitReal(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	arg := b.emitExpr(ctx, expr.Args[0])[0]
	T := b.GetStoredType(ctx, b.typeOf(expr))
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
	lenValue := b.emitTypeConversion(ctx, args[1], b.typeOf(expr.Args[1]), types.Typ[types.Uintptr], location)

	// Add the values.
	op = mlir.GoCreateAddIOperation(b.ctx, b.uiptr, ptrValue, lenValue, location)
	appendOperation(ctx, op)
	resultValue := resultOf(op)

	// Convert the integer back into a pointer value and then return the result.
	op = mlir.GoCreateIntToPtrOperation(b.ctx, resultValue, b.ptr, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitUnsafeSizeof(ctx context.Context, expr *ast.CallExpr) []mlir.Value {
	location := b.location(expr.Pos())
	T := b.GetStoredType(ctx, b.typeOf(expr.Args[0]))
	sz := mlir.GoGetTypeSizeInBytes(T, b.config.Module)
	return []mlir.Value{b.emitConstInt(ctx, int64(sz), b.uiptr, location)}
}
