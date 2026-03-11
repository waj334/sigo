package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitArith(ctx context.Context, op token.Token, X, Y mlir.ValueLike, XT types.Type, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	var result mlir.ValueLike

	// Create the respective binary expression operation.
	switch op {
	case token.ADD:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewAddIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsString):
			op := goir.NewAddStrOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsFloat):
			op := goir.NewAddFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsComplex):
			op := goir.NewAddCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	case token.SUB:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewSubIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsFloat):
			op := goir.NewSubFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsComplex):
			op := goir.NewSubCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	case token.MUL:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewMulIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsFloat):
			op := goir.NewMulFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsComplex):
			op := goir.NewMulCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	case token.QUO:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := goir.NewDivUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewDivSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsFloat):
			op := goir.NewDivFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsComplex):
			op := goir.NewDivCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	case token.REM:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := goir.NewRemUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewRemSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsFloat):
			op := goir.NewRemFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	case token.AND:
		op := goir.NewAndOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = op.Result(0)
	case token.OR:
		op := goir.NewOrOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = op.Result(0)
	case token.XOR:
		op := goir.NewXorOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = op.Result(0)
	case token.AND_NOT:
		op := goir.NewAndNotOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = op.Result(0)
	case token.SHL:
		op := goir.NewShlOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = op.Result(0)
	case token.SHR:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := goir.NewShrUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		case typeHasFlags(XT, types.IsInteger):
			op := goir.NewShrSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = op.Result(0)
		}
	default:
		panic("unhandled arithmetic operation")
	}

	if result == nil {
		panic("operation yielded no result")
	}

	return result.AsValue()
}

func (b *Builder) emitIncDecStatement(ctx context.Context, stmt *ast.IncDecStmt) {
	var result mlir.ValueLike
	location := b.location(ctx, stmt.Pos())

	// Get address of the LHS value to store the result at later.
	var value Value
	switch X := stmt.X.(type) {
	case *ast.Ident:
		value = b.valueOf(ctx, X)
	case *ast.SelectorExpr:
		value = b.NewTempValue(b.emitSelectAddr(ctx, X))
	default:
		panic("unhandled")
	}

	// Evaluate the LHS.
	lhs := value.Load(ctx, location)

	// Create the respective constant one value matching the LHS type.
	T := b.typeOf(ctx, stmt.X).Underlying()
	intType := b.GetStoredType(ctx, T)
	constOneOp := goir.NewConstantOperation(b.ctx, b.intAttr(1), mlir.NewNullAttribute(), intType, location)
	constOne := resultOf(constOneOp)
	appendOperation(ctx, constOneOp)

	// Perform the respective arithmetic.
	switch stmt.Tok {
	case token.INC:
		// Add one to the lhs value.
		op := goir.NewAddIOperation(b.ctx, intType, lhs, constOne, location)
		appendOperation(ctx, op)
		result = resultOf(op)
	case token.DEC:
		// Subtract one to the lhs value.
		op := goir.NewSubIOperation(b.ctx, intType, lhs, constOne, location)
		appendOperation(ctx, op)
		result = resultOf(op)
	default:
		panic("unhandled increment/decrement statement")
	}

	// Store the value at the address of the LHS.
	value.Store(ctx, result, location)
}
