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
