package ssa

import (
	"context"

	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) cmpIPredicate(tok token.Token, unsigned bool) mlir.AttributeLike {
	switch tok {
	case token.EQL:
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateEq)
	case token.GEQ:
		if unsigned {
			return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateUge)
		}
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateSge)
	case token.GTR:
		if unsigned {
			return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateUgt)
		}
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateSgt)
	case token.LEQ:
		if unsigned {
			return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateUle)
		}
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateSle)
	case token.LSS:
		if unsigned {
			return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateUlt)
		}
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateSlt)
	case token.NEQ:
		return goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateNe)
	default:
		panic("invalid integer comparison predicate")
	}
}

func (b *Builder) cmpFPredicate(tok token.Token) mlir.Attribute {
	switch tok {
	case token.EQL:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateEq)
	case token.GEQ:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateGe)
	case token.GTR:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateGt)
	case token.LEQ:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateLe)
	case token.LSS:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateLt)
	case token.NEQ:
		return goir.NewCmpFPredicateAttr(b.ctx, goir.CmpFPredicateNe)
	default:
		panic("invalid float comparison predicate")
	}
}

func (b *Builder) emitIntegerCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	baseT := goir.GetBaseType(X.Type())
	unsigned := true
	if intT, ok := goir.AsIntegerType(baseT); ok {
		unsigned = isUnsigned(intT)
	}
	predicate := b.cmpIPredicate(op, unsigned)
	cmpOp := goir.NewCmpIOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp).AsValue()
}

func (b *Builder) emitFloatCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	predicate := b.cmpFPredicate(op)
	cmpOp := goir.NewCmpFOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp).AsValue()
}

func (b *Builder) emitComplexCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Get the operand values to be used in the binary expression.
	predicate := b.cmpFPredicate(op)
	cmpOp := goir.NewCmpCOperation(b.ctx, b.i1, predicate, X, Y, location)
	appendOperation(ctx, cmpOp)
	return resultOf(cmpOp).AsValue()
}

func (b *Builder) emitInterfaceCompare(ctx context.Context, predicate token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	op := goir.NewCmpInterfaceOperation(b.ctx, b.i1, X, Y, location)
	appendOperation(ctx, op)
	result := resultOf(op).AsValue()

	if predicate == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}

	return result
}

func (b *Builder) emitStringCompare(ctx context.Context, predicate token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	switch predicate {
	case token.EQL:
		op := goir.NewCmpStringOperation(b.ctx, b.i1, goir.NewCmpPredicateAttr(b.ctx, goir.CmpPredicateEq), X, Y, location)
		appendOperation(ctx, op)
		return resultOf(op).AsValue()
	case token.NEQ:
		op := goir.NewCmpStringOperation(b.ctx, b.i1, goir.NewCmpPredicateAttr(b.ctx, goir.CmpPredicateNe), X, Y, location)
		appendOperation(ctx, op)
		return resultOf(op).AsValue()
	default:
		panic("unhandled comparison predicate type")
	}
}

func (b *Builder) emitPointerCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Reinterpret pointers to integers.
	X = b.emitCastPointerToInt(ctx, X, location)
	Y = b.emitCastPointerToInt(ctx, Y, location)

	// Compare the integer values.
	cmpOp := goir.NewCmpIOperation(b.ctx, b.i1, goir.NewCmpIPredicateAttr(b.ctx, goir.CmpIPredicateEq), X, Y, location)
	appendOperation(ctx, cmpOp)
	result := resultOf(cmpOp).AsValue()
	if op == token.NEQ {
		// Negate the result.
		result = b.emitNegation(ctx, result, location)
	}
	return result
}

func (b *Builder) emitFuncCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Convert the functions to pointers.
	funcToPtrOp := goir.NewFunctionToPointerOperation(b.ctx, X, b.ptr, location)
	appendOperation(ctx, funcToPtrOp)
	X = resultOf(funcToPtrOp)

	funcToPtrOp = goir.NewFunctionToPointerOperation(b.ctx, Y, b.ptr, location)
	appendOperation(ctx, funcToPtrOp)
	Y = resultOf(funcToPtrOp)

	// Compare as pointers.
	return b.emitPointerCompare(ctx, op, X, Y, location)
}

func (b *Builder) emitStructCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, T *types.Struct, location mlir.LocationLike) mlir.Value {
	// Start with true (assuming structs are equal).
	result := b.emitConstBool(ctx, true, b.i1, location)

	// Compare each field, AND-ing results together in a single block.
	for i := 0; i < T.NumFields(); i++ {
		field := T.Field(i)
		elementType := b.GetStoredType(ctx, field.Type())

		// Extract the struct fields at the current index.
		extractOp := goir.NewExtractOperation(b.ctx, uint64(i), elementType, X, location)
		appendOperation(ctx, extractOp)
		Xi := resultOf(extractOp).AsValue()

		extractOp = goir.NewExtractOperation(b.ctx, uint64(i), elementType, Y, location)
		appendOperation(ctx, extractOp)
		Yi := resultOf(extractOp).AsValue()

		// Compare the values for equality.
		var cond mlir.Value
		switch {
		case typeHasFlags(field.Type(), types.IsBoolean), typeHasFlags(field.Type(), types.IsInteger):
			cond = b.emitIntegerCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(field.Type(), types.IsFloat):
			cond = b.emitFloatCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(field.Type(), types.IsComplex):
			cond = b.emitComplexCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(field.Type(), types.IsString):
			cond = b.emitStringCompare(ctx, token.EQL, Xi, Yi, location)
		case isPointer(field.Type()):
			cond = b.emitPointerCompare(ctx, token.EQL, Xi, Yi, location)
		case typeIs[*types.Interface](field.Type()):
			cond = b.emitInterfaceCompare(ctx, token.EQL, Xi, Yi, location)
		case typeIs[*types.Signature](field.Type()):
			cond = b.emitFuncCompare(ctx, token.EQL, Xi, Yi, location)
		case typeIs[*types.Struct](field.Type()):
			cond = b.emitStructCompare(ctx, token.EQL, Xi, Yi, field.Type().Underlying().(*types.Struct), location)
		case typeIs[*types.Array](field.Type()):
			cond = b.emitArrayCompare(ctx, token.EQL, Xi, Yi, field.Type().Underlying().(*types.Array), location)
		default:
			panic("unhandled switch comparison operand type")
		}

		// AND the field comparison result with the running result.
		andOp := goir.NewAndOperation(b.ctx, b.i1, result, cond, location)
		appendOperation(ctx, andOp)
		result = resultOf(andOp).AsValue()
	}

	if op == token.NEQ {
		result = b.emitNegation(ctx, result, location)
	}

	return result
}

func (b *Builder) emitArrayCompare(ctx context.Context, op token.Token, X mlir.ValueLike, Y mlir.ValueLike, T *types.Array, location mlir.LocationLike) mlir.Value {
	elemType := T.Elem()
	elementT := b.GetStoredType(ctx, elemType)

	// Start with true (assuming arrays are equal).
	result := b.emitConstBool(ctx, true, b.i1, location)

	// Compare each element, AND-ing results together in a single block.
	for i := int64(0); i < T.Len(); i++ {
		// Extract the array elements at the current index.
		extractOp := goir.NewExtractOperation(b.ctx, uint64(i), elementT, X, location)
		appendOperation(ctx, extractOp)
		Xi := resultOf(extractOp).AsValue()

		extractOp = goir.NewExtractOperation(b.ctx, uint64(i), elementT, Y, location)
		appendOperation(ctx, extractOp)
		Yi := resultOf(extractOp).AsValue()

		// Compare the values for equality.
		var cond mlir.Value
		switch {
		case typeHasFlags(elemType, types.IsBoolean), typeHasFlags(elemType, types.IsInteger):
			cond = b.emitIntegerCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(elemType, types.IsFloat):
			cond = b.emitFloatCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(elemType, types.IsComplex):
			cond = b.emitComplexCompare(ctx, token.EQL, Xi, Yi, location)
		case typeHasFlags(elemType, types.IsString):
			cond = b.emitStringCompare(ctx, token.EQL, Xi, Yi, location)
		case isPointer(elemType):
			cond = b.emitPointerCompare(ctx, token.EQL, Xi, Yi, location)
		case typeIs[*types.Interface](elemType):
			cond = b.emitInterfaceCompare(ctx, token.EQL, Xi, Yi, location)
		case typeIs[*types.Struct](elemType):
			cond = b.emitStructCompare(ctx, token.EQL, Xi, Yi, elemType.Underlying().(*types.Struct), location)
		case typeIs[*types.Array](elemType):
			cond = b.emitArrayCompare(ctx, token.EQL, Xi, Yi, elemType.Underlying().(*types.Array), location)
		default:
			panic("unhandled array element comparison operand type")
		}

		// AND the element comparison result with the running result.
		andOp := goir.NewAndOperation(b.ctx, b.i1, result, cond, location)
		appendOperation(ctx, andOp)
		result = resultOf(andOp).AsValue()
	}

	if op == token.NEQ {
		result = b.emitNegation(ctx, result, location)
	}

	return result
}

func (b *Builder) emitComparison(ctx context.Context, expr *ast.BinaryExpr) mlir.Value {
	X := b.emitExpr(ctx, expr.X)[0]
	location := b.location(ctx, expr.OpPos)

	XT := baseType(resolveType(ctx, b.typeOf(ctx, expr.X)))
	YT := baseType(resolveType(ctx, b.typeOf(ctx, expr.Y)))

	switch {
	case typeHasFlags(XT, types.IsBoolean), typeHasFlags(XT, types.IsInteger):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitIntegerCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsFloat):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitFloatCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsComplex):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitComplexCompare(ctx, expr.Op, X, Y, location)
	case typeHasFlags(XT, types.IsString):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitStringCompare(ctx, expr.Op, X, Y, location)
	case isPointer(XT):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
	case typeIs[*types.Interface](XT):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
			YT = XT
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitInterfaceCompare(ctx, expr.Op, X, Y, location)
	case typeIs[*types.Struct](XT):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitStructCompare(ctx, expr.Op, X, Y, XT.Underlying().(*types.Struct), location)
	case typeIs[*types.Array](XT):
		var Y mlir.ValueLike
		if isNil(YT) {
			Y = b.emitZeroValue(ctx, XT, location)
		} else {
			Y = b.emitExpr(ctx, expr.Y)[0]
		}
		return b.emitArrayCompare(ctx, expr.Op, X, Y, XT.Underlying().(*types.Array), location)

		// The following are special cases when compared against nil:
	case typeIs[*types.Signature](XT):
		// X is either a function pointer or a struct representing a struct. Create the respective zero value to compare
		// against.

		zeroOp := goir.NewZeroOperation(b.ctx, b.ptr, location)
		appendOperation(ctx, zeroOp)
		Y := resultOf(zeroOp).AsValue()

		if goir.TypeIsAFunctionType(X.Type()) {
			// Compare as pointers.
			funcToPtrOp := goir.NewFunctionToPointerOperation(b.ctx, X, b.ptr, location)
			appendOperation(ctx, funcToPtrOp)
			X = resultOf(funcToPtrOp).AsValue()
			return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
		}

		// Compare the function pointer struct member to nullptr.
		extractOp := goir.NewExtractOperation(b.ctx, 0, b.ptr, X, location)
		appendOperation(ctx, extractOp)
		X = resultOf(extractOp).AsValue()
		return b.emitPointerCompare(ctx, expr.Op, X, Y, location)
	case typeIs[*types.Chan](XT), typeIs[*types.Slice](XT), typeIs[*types.Map](XT):
		op := goir.NewCmpNilOperation(b.ctx, b.i1, X, location)
		appendOperation(ctx, op)
		value := resultOf(op).AsValue()
		if expr.Op == token.NEQ {
			// Negate the result.
			value = b.emitNegation(ctx, value, location)
		}

		return value
	default:
		panic("unhandled comparison operation")
	}
}

func (b *Builder) emitNegation(ctx context.Context, X mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Negate the input boolean value.
	constTrueOp := goir.NewConstantOperation(b.ctx, b.boolAttr(true), nil, b.i1, location)
	appendOperation(ctx, constTrueOp)

	xorOp := goir.NewXorOperation(b.ctx, b.i1, X, resultOf(constTrueOp).AsValue(), location)
	appendOperation(ctx, xorOp)
	return resultOf(xorOp).AsValue()
}

func (b *Builder) emitLogicalComparison(ctx context.Context, expr *ast.BinaryExpr) mlir.Value {
	location := b.location(ctx, expr.Pos())

	// Create the exit block where execution should continue following the expression.
	exitBlock := mlir.NewBlock([]mlir.TypeLike{b.i1}, []mlir.LocationLike{location})

	// Evaluate X the current block.
	X := b.emitExpr(ctx, expr.X)[0]

	// Create the block in which to evaluate Y.
	yBlock := mlir.NewBlock(nil, nil)
	buildBlock(ctx, yBlock, func() {
		// Evaluate Y in the other block.
		Y := b.emitExpr(ctx, expr.Y)[0]

		// The result of the above expression is the result of the entire logical comparison.
		brOp := goir.NewBranchOperation(b.ctx, exitBlock, []mlir.ValueLike{Y}, location)
		appendOperation(ctx, brOp)
	})
	appendBlock(ctx, yBlock)

	switch expr.Op {
	case token.LAND:
		// Branch to exit block passing either Y (X = true) or X (X = false) as the block parameter.
		condBrOp := goir.NewCondBranchOperation(b.ctx, X, yBlock, []mlir.ValueLike{}, exitBlock, []mlir.ValueLike{X},
			location)
		appendOperation(ctx, condBrOp)
	case token.LOR:
		// Branch to exit block passing either X (X = true) or Y (X = false) as the block parameter.
		condBrOp := goir.NewCondBranchOperation(b.ctx, X, exitBlock, []mlir.ValueLike{X}, yBlock, []mlir.ValueLike{},
			location)
		appendOperation(ctx, condBrOp)
	default:
		panic("invalid logical comparison")
	}

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block and return the block parameter.
	setCurrentBlock(ctx, exitBlock)
	return exitBlock.Argument(0).AsValue()
}
