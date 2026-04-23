package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitBuiltinCall(ctx context.Context, expr *ast.CallExpr) []mlir.ValueLike {
	location := b.location(ctx, expr.Pos())
	anyType := types.NewInterfaceType(nil, nil)

	var signature *types.Signature
	switch T := b.typeOf(ctx, expr.Fun).(type) {
	case *types.Signature:
		signature = T
	default:
		// Create a synthetic signature.
		resultType := b.typeOf(ctx, expr)
		resultTuple := types.NewTuple(types.NewVar(token.NoPos, nil, "result", resultType))

		inputs := make([]*types.Var, len(expr.Args))
		for i, arg := range expr.Args {
			argType := b.typeOf(ctx, arg)
			if isUntyped(argType) {
				argType = types.Default(argType)
			}
			inputs[i] = types.NewVar(token.NoPos, nil, fmt.Sprintf("param$%d", i), argType)
		}
		paramsTuple := types.NewTuple(inputs...)
		signature = types.NewSignatureType(nil, nil, nil, paramsTuple, resultTuple, false)
	}

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

	var results []mlir.TypeLike

	offset := 0
	switch name {
	case "new", "make":
		// First argument is a type. Unwrap named types so the verifier sees
		// the underlying map/slice/chan type.
		results = append(results, b.GetStoredType(ctx, baseType(b.typeOf(ctx, expr))))
		offset = 1
	case "panic":
		valueType := b.typeOf(ctx, expr.Args[0])
		value := b.emitExpr(ctx, expr.Args[0])[0]
		value = b.emitInterfaceValue(ctx, anyType, valueType, value, location)
		op := goir.NewPanicOperation(b.ctx, value, location)
		appendOperation(ctx, op)
		return nil
	case "recover":
		op := goir.NewRecoverOperation(b.ctx, b._any, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		resultType := b.typeOf(ctx, expr)
		if resultType != nil {
			switch resultType := resultType.(type) {
			case *types.Tuple:
				// Do nothing.
			default:
				results = []mlir.TypeLike{b.GetStoredType(ctx, resultType)}
			}
		}
	}

	// Emit argument values.
	// TODO: The logic below could probably be simplified.
	var operands []mlir.ValueLike
	if signature.Variadic() {
		operands = b.emitCallArgs(ctx, signature, expr)
	} else {
		for _, argExpr := range expr.Args[offset:] {
			operands = append(operands, b.emitExpr(ctx, argExpr)...)
		}

		switch name {
		case "make":
			for i := range operands {
				// Convert to integer type.
				// TODO: This should be done during IR lowering.
				operands[i] = b.emitTypeConversion(ctx, operands[i], b.typeOf(ctx, expr.Args[offset+i]), types.Typ[types.Int], location)
			}
		default:
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
	}

	// Finally, emit the built-in call.
	op := goir.NewBuiltInCallOperation(b.ctx, name, results, operands, location)
	appendOperation(ctx, op)
	callResults := resultsOf(op)

	// For make/new: the builtin result uses the underlying type (map/slice/chan),
	// but if the Go type is a named type, we need to bitcast back so that stores
	// to the named-type variable match.
	if name == "make" || name == "new" {
		goType := b.typeOf(ctx, expr)
		if _, isNamed := goType.(*types.Named); isNamed {
			namedT := b.GetStoredType(ctx, goType)
			for i := range callResults {
				callResults[i] = b.bitcastTo(ctx, callResults[i], namedT, location)
			}
		}
	}

	return callResults
}
