package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitArith(ctx context.Context, op token.Token, X, Y mlir.Value, XT types.Type, T mlir.Type, location mlir.Location) mlir.Value {
	var result mlir.Value

	// Create the respective binary expression operation.
	switch op {
	case token.ADD:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateAddIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsString):
			op := mlir.GoCreateAddStrOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsFloat):
			op := mlir.GoCreateAddFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsComplex):
			op := mlir.GoCreateAddCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	case token.SUB:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateSubIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsFloat):
			op := mlir.GoCreateSubFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsComplex):
			op := mlir.GoCreateSubCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	case token.MUL:
		switch {
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateMulIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsFloat):
			op := mlir.GoCreateMulFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsComplex):
			op := mlir.GoCreateMulCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	case token.QUO:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := mlir.GoCreateDivUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateDivSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsFloat):
			op := mlir.GoCreateDivFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsComplex):
			op := mlir.GoCreateDivCOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	case token.REM:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := mlir.GoCreateRemUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateRemSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsFloat):
			op := mlir.GoCreateRemFOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	case token.AND:
		op := mlir.GoCreateAndOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = mlir.OperationGetResult(op, 0)
	case token.OR:
		op := mlir.GoCreateOrOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = mlir.OperationGetResult(op, 0)
	case token.XOR:
		op := mlir.GoCreateXorOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = mlir.OperationGetResult(op, 0)
	case token.AND_NOT:
		op := mlir.GoCreateAndNotOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = mlir.OperationGetResult(op, 0)
	case token.SHL:
		op := mlir.GoCreateShlOperation(b.ctx, T, X, Y, location)
		appendOperation(ctx, op)
		result = mlir.OperationGetResult(op, 0)
	case token.SHR:
		switch {
		case typeHasFlags(XT, types.IsInteger, types.IsUnsigned):
			op := mlir.GoCreateShrUIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		case typeHasFlags(XT, types.IsInteger):
			op := mlir.GoCreateShrSIOperation(b.ctx, T, X, Y, location)
			appendOperation(ctx, op)
			result = mlir.OperationGetResult(op, 0)
		}
	default:
		panic("unhandled arithmetic operation")
	}

	if result == nil {
		panic("operation yielded no result")
	}

	return result
}

func (b *Builder) emitIncDecStatement(ctx context.Context, stmt *ast.IncDecStmt) {
	var result mlir.Value
	location := b.location(stmt.Pos())

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
	T := b.typeOf(stmt.X).Underlying()
	intType := b.GetStoredType(ctx, T)
	intAttrType := mlir.IntegerTypeGet(b.ctx, 64)
	constOneOp := mlir.GoCreateConstantOperation(b.ctx, mlir.IntegerAttrGet(intAttrType, 1), intType, location)
	constOne := resultOf(constOneOp)
	appendOperation(ctx, constOneOp)

	// Perform the respective arithmetic.
	switch stmt.Tok {
	case token.INC:
		// Add one to the lhs value.
		op := mlir.GoCreateAddIOperation(b.ctx, intType, lhs, constOne, location)
		appendOperation(ctx, op)
		result = resultOf(op)
	case token.DEC:
		// Subtract one to the lhs value.
		op := mlir.GoCreateSubIOperation(b.ctx, intType, lhs, constOne, location)
		appendOperation(ctx, op)
		result = resultOf(op)
	default:
		panic("unhandled increment/decrement statement")
	}

	// Store the value at the address of the LHS.
	value.Store(ctx, result, location)
}
