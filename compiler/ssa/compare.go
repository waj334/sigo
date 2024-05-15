package ssa

import (
	"context"

	"go/ast"
	"go/token"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) cmpIPredicate(tok token.Token, unsigned bool) mlir.Attribute {
	switch tok {
	case token.EQL:
		return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_eq)
	case token.GEQ:
		if unsigned {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_uge)
		} else {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_sge)
		}
	case token.GTR:
		if unsigned {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_ugt)
		} else {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_sgt)
		}
	case token.LEQ:
		if unsigned {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_ule)
		} else {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_sle)
		}
	case token.LSS:
		if unsigned {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_ult)
		} else {
			return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_slt)
		}
	case token.NEQ:
		return mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_ne)
	default:
		panic("invalid integer comparison predicate")
	}
}

func (b *Builder) cmpFPredicate(tok token.Token) mlir.Attribute {
	switch tok {
	case token.EQL:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_eq)
	case token.GEQ:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_ge)
	case token.GTR:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_gt)
	case token.LEQ:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_le)
	case token.LSS:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_lt)
	case token.NEQ:
		return mlir.GoCreateCmpFPredicate(b.ctx, mlir.GoCmpFPredicate_ne)
	default:
		panic("invalid float comparison predicate")
	}
}

func (b *Builder) emitIntegerCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	baseType := mlir.GoGetBaseType(mlir.ValueGetType(X))
	predicate := b.cmpIPredicate(op, isUnsigned(baseType))
	cmpOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp)
}

func (b *Builder) emitFloatCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	predicate := b.cmpFPredicate(op)
	cmpOp := mlir.GoCreateCmpFOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp)
}

func (b *Builder) emitComplexCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	// Get the operand values to be used in the binary expression.
	predicate := b.cmpFPredicate(op)
	cmpOp := mlir.GoCreateCmpFOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp)
}

func (b *Builder) emitInterfaceCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, YT types.Type, location mlir.Location) mlir.Value {
	var result mlir.Value
	if typeIs[*types.Interface](YT) {
		// Emit the runtime call to compare the two interface types.
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.interfaceCompare", b.types(b.i1), b.values(X, Y), location)
		appendOperation(ctx, op)
		result = resultOf(op)
	} else {
		typeInfoOp := mlir.GoCreateTypeInfoOperation(b.ctx, b.typeInfoPtr, b.GetType(ctx, YT), location)
		appendOperation(ctx, typeInfoOp)
		infoValue := resultOf(typeInfoOp)

		// Take the address of the RHS value.
		Y = b.makeCopyOf(ctx, Y, location)

		// Reinterpret the pointer as an unsafe.Pointer.
		Y = b.bitcastTo(ctx, Y, b.ptr, location)

		// Emit the runtime call to compare the interface type against the arbitrary value.
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.interfaceCompareTo", b.types(b.i1), b.values(X, infoValue, Y), location)
		appendOperation(ctx, op)
		result = resultOf(op)
	}

	if op == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}

	return result
}

func (b *Builder) emitStringCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	cmpOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringCompare", b.types(b.i1), b.values(X, Y), location)
	appendOperation(ctx, cmpOp)
	result := resultOf(cmpOp)
	if op == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}
	return result
}

func (b *Builder) emitPointerCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	// Reinterpret pointers to integers.
	X = b.emitCastPointerToInt(ctx, X, location)
	Y = b.emitCastPointerToInt(ctx, Y, location)

	// Compare the integer values.
	cmpOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, mlir.GoCreateCmpIPredicate(b.ctx, mlir.GoCmpIPredicate_eq), X, Y, location)
	appendOperation(ctx, cmpOp)
	result := resultOf(cmpOp)
	if op == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}
	return result
}

func (b *Builder) emitFuncCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, location mlir.Location) mlir.Value {
	// Convert the functions to pointers.
	funcToPtrOp := mlir.GoCreateFunctionToPointerOperation(b.ctx, X, b.ptr, location)
	appendOperation(ctx, funcToPtrOp)
	X = resultOf(funcToPtrOp)

	funcToPtrOp = mlir.GoCreateFunctionToPointerOperation(b.ctx, Y, b.ptr, location)
	appendOperation(ctx, funcToPtrOp)
	Y = resultOf(funcToPtrOp)

	// Compare as pointers.
	return b.emitPointerCompare(ctx, op, X, Y, location)
}

func (b *Builder) emitStructCompare(ctx context.Context, op token.Token, X mlir.Value, Y mlir.Value, T *types.Struct, location mlir.Location) mlir.Value {
	successor := mlir.BlockCreate2([]mlir.Type{b.i1}, []mlir.Location{location})

	// Compare each field until one does not match.
	for i := 0; i < T.NumFields(); i++ {
		field := T.Field(i)
		elementType := b.GetStoredType(ctx, field.Type())

		// Extract the struct fields at the current index.
		extractOp := mlir.GoCreateExtractOperation(b.ctx, uint64(i), elementType, X, location)
		appendOperation(ctx, extractOp)
		X := resultOf(extractOp)

		extractOp = mlir.GoCreateExtractOperation(b.ctx, uint64(i), elementType, Y, location)
		appendOperation(ctx, extractOp)
		Y := resultOf(extractOp)

		// Compare the values for equality.
		var cond mlir.Value
		switch {
		case typeHasFlags(field.Type(), types.IsBoolean), typeHasFlags(field.Type(), types.IsInteger):
			cond = b.emitIntegerCompare(ctx, token.EQL, X, Y, location)
		case typeHasFlags(field.Type(), types.IsFloat):
			cond = b.emitFloatCompare(ctx, token.EQL, X, Y, location)
		case typeHasFlags(field.Type(), types.IsComplex):
			cond = b.emitComplexCompare(ctx, token.EQL, X, Y, location)
		case typeHasFlags(field.Type(), types.IsString):
			cond = b.emitStringCompare(ctx, token.EQL, X, Y, location)
		case isPointer(field.Type()):
			cond = b.emitPointerCompare(ctx, token.EQL, X, Y, location)
		case typeIs[*types.Interface](field.Type()):
			cond = b.emitInterfaceCompare(ctx, token.EQL, X, Y, field.Type().Underlying(), location)
		case typeIs[*types.Signature](field.Type()):
			return b.emitFuncCompare(ctx, token.EQL, X, Y, location)
		case typeIs[*types.Struct](field.Type()):
			cond = b.emitStructCompare(ctx, token.EQL, X, Y, field.Type().Underlying().(*types.Struct), location)
		default:
			panic("unhandled switch comparison operand type")
		}

		// Create the next block that the next compare will be emitted into.
		nextBlock := mlir.BlockCreate2(nil, nil)

		// Conditionally branch to the successor block passing false if the fields don't match. Otherwise, branch to the
		// next block in order to evaluate the comparison of the next field.
		falseValue := b.emitConstBool(ctx, false, location)
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, cond, nextBlock, nil, successor, []mlir.Value{falseValue}, location)
		appendOperation(ctx, condBrOp)

		// Continue emission in the next block.
		appendBlock(ctx, nextBlock)
		setCurrentBlock(ctx, nextBlock)
	}

	// Branch to the successor block passing true.
	trueValue := b.emitConstBool(ctx, true, location)
	brOp := mlir.GoCreateBranchOperation(b.ctx, successor, []mlir.Value{trueValue}, location)
	appendOperation(ctx, brOp)

	// Continue emission into the successor block.
	appendBlock(ctx, successor)
	setCurrentBlock(ctx, successor)

	result := mlir.BlockGetArgument(successor, 0)
	if op == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}

	// Result the successor block argument.
	return result
}

func (b *Builder) emitComparison(ctx context.Context, expr *ast.BinaryExpr) mlir.Value {
	X := b.emitExpr(ctx, expr.X)[0]
	location := b.location(expr.Pos())

	XT := baseType(b.typeOf(ctx, expr.X))
	YT := baseType(b.typeOf(ctx, expr.Y))

	/*
		if typeHasFlags(YT, types.IsUntyped) {
			// Infer the type to compare against.
			lhsTypes := currentLhsList(ctx)
			index := currentRhsIndex(ctx)
			YT = lhsTypes[index]
		}
	*/

	switch {
	case typeHasFlags(XT, types.IsBoolean), typeHasFlags(XT, types.IsInteger):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitIntegerCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsFloat):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitFloatCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsComplex):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitComplexCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsString):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitStringCompare(ctx, expr.Op, X, Y, location)
	case isPointer(b.typeOf(ctx, expr.X)):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
	case typeIs[*types.Interface](XT):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
			YT = XT
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitInterfaceCompare(ctx, expr.Op, X, Y, YT, location)
	case typeIs[*types.Struct](XT):
		var Y mlir.Value
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitStructCompare(ctx, token.EQL, X, Y, XT.Underlying().(*types.Struct), location)

		// The following are special cases when compared against nil:
	case typeIs[*types.Signature](XT):
		// X is either a function pointer or a struct representing a struct. Create the respective zero value to compare
		// against.

		zeroOp := mlir.GoCreateZeroOperation(b.ctx, b.ptr, location)
		appendOperation(ctx, zeroOp)
		Y := resultOf(zeroOp)

		if mlir.TypeIsAFunction(mlir.ValueGetType(X)) {
			// Compare as pointers.
			funcToPtrOp := mlir.GoCreateFunctionToPointerOperation(b.ctx, X, b.ptr, location)
			appendOperation(ctx, funcToPtrOp)
			X = resultOf(funcToPtrOp)
			return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
		}

		// Compare the function pointer struct member to nullptr.
		extractOp := mlir.GoCreateExtractOperation(b.ctx, 0, b.ptr, X, location)
		appendOperation(ctx, extractOp)
		X = resultOf(extractOp)
		return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
	case typeIs[*types.Slice](XT):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.sliceIsNil", b.types(b.i1), b.values(X), location)
		appendOperation(ctx, op)
		return resultOf(op)
	case typeIs[*types.Map](XT):
		op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.mapIsNil", b.types(b.i1), b.values(X), location)
		appendOperation(ctx, op)
		return resultOf(op)
	default:
		panic("unhandled comparison operation")
	}
}

func (b *Builder) emitNegation(ctx context.Context, X mlir.Value, location mlir.Location) mlir.Value {
	// Negate the input boolean value.
	trueAttr := mlir.IntegerAttrGet(b.i1, 1)
	constTrueOp := mlir.GoCreateConstantOperation(b.ctx, trueAttr, b.i1, location)
	appendOperation(ctx, constTrueOp)

	xorOp := mlir.GoCreateXorOperation(b.ctx, b.i1, X, resultOf(constTrueOp), location)
	appendOperation(ctx, xorOp)
	return resultOf(xorOp)
}

func (b *Builder) emitLogicalComparison(ctx context.Context, expr *ast.BinaryExpr) mlir.Value {
	location := b.location(expr.Pos())

	// Create the exit block where execution should continue following the expression.
	exitBlock := mlir.BlockCreate2([]mlir.Type{b.i1}, []mlir.Location{location})

	// Evaluate X amnd Y in the current block.
	X := b.emitExpr(ctx, expr.X)[0]
	Y := b.emitExpr(ctx, expr.Y)[0]

	switch expr.Op {
	case token.LAND:
		// Branch to exit block passing either Y (X = true) or X (X = false) as the block parameter.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, X, exitBlock, []mlir.Value{Y}, exitBlock, []mlir.Value{X},
			location)
		appendOperation(ctx, condBrOp)
	case token.LOR:
		// Branch to exit block passing either X (X = true) or Y (X = false) as the block parameter.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, X, exitBlock, []mlir.Value{X}, exitBlock, []mlir.Value{Y},
			location)
		appendOperation(ctx, condBrOp)
	default:
		panic("invalid logical comparison")
	}

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block and return the block parameter.
	setCurrentBlock(ctx, exitBlock)
	return mlir.BlockGetArgument(exitBlock, 0)
}
