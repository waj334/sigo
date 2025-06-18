package ssa

import (
	"context"
	"go/ast"
	"go/constant"
	"go/types"

	"pkg.si-go.dev/sigo/mlir"
)

func (b *Builder) emitConstantDecl(ctx context.Context, decl *ast.GenDecl) {
	for _, spec := range decl.Specs {
		valueSpec := spec.(*ast.ValueSpec)
		for i, ident := range valueSpec.Names {
			location := b.location(ident.Pos())
			obj := b.objectOf(ctx, ident).(*types.Const)
			constT := b.typeOf(ctx, ident)
			if typeHasFlags(constT, types.IsUntyped) {
				constT = types.Default(constT)
			}
			T := b.GetStoredType(ctx, constT)

			fromObj := func(obj *types.Const) mlir.Attribute {
				litType := types.Unalias(constT).Underlying().(*types.Basic)
				switch litType.Kind() {
				case types.Bool:
					return b.boolAttr(constant.BoolVal(obj.Val()))
				case types.Complex64:
					realValue, _ := constant.Float32Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float32Val(constant.Imag(obj.Val()))
					return mlir.GoCreateComplexNumberAttr(b.ctx, b.f32, float64(realValue), float64(imagValue))
				case types.Complex128:
					realValue, _ := constant.Float64Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float64Val(constant.Imag(obj.Val()))
					return mlir.GoCreateComplexNumberAttr(b.ctx, b.f64, realValue, imagValue)
				case types.Float32:
					k, _ := constant.Float32Val(obj.Val())
					return mlir.FloatAttrDoubleGet(b.ctx, b.f32, float64(k))
				case types.Float64:
					k, _ := constant.Float64Val(obj.Val())
					return mlir.FloatAttrDoubleGet(b.ctx, b.f32, k)
				case types.Int, types.Int8, types.Int16, types.Int32, types.Int64:
					constVal, _ := constant.Int64Val(obj.Val())
					return b.intAttr(constVal)
				case types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.Uintptr:
					constVal, _ := constant.Uint64Val(obj.Val())
					return b.intAttr(int64(constVal))
				case types.String:
					return b.strAttr(constant.StringVal(obj.Val()))
				default:
					panic("unreachable: " + litType.String())
				}
			}

			var body mlir.Block
			var value mlir.Attribute
			if len(valueSpec.Values) > 0 {
				switch expr := valueSpec.Values[i].(type) {
				case *ast.BasicLit:
					value = fromObj(obj)
				default:
					// Emit the expression into the body block and then yield the result.
					body = mlir.BlockCreate2(nil, nil)
					ctx := newContextWithCurrentBlock(ctx)
					setCurrentBlock(ctx, body)
					result := b.emitExpr(ctx, expr)[0]

					// Create the yield operation.
					yieldOp := mlir.GoCreateYieldOperation(b.ctx, result, location)
					appendOperation(ctx, yieldOp)
				}
			} else {
				value = fromObj(obj)
			}

			symbolName := qualifiedName(obj.Name(), obj.Pkg())

			// Is this constant at the global scope?
			if obj.Parent() == obj.Pkg().Scope() {
				// Emit a global constant that will be referred to later when used.
				// NOTE: value is nil if the constant value was NOT a literal value.
				constOp := mlir.GoCreateGlobalConstantOperation(b.ctx, value, b.strAttr(symbolName), location)
				if body != nil {
					mlir.GoGlobalConstantOperationAddBody(constOp, body)
				}

				appendOperation(ctx, constOp)
			}

			b.setAddr(ctx, ident, ConstantValue{
				Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
					// Is this constant at the global scope?
					if obj.Parent() == obj.Pkg().Scope() {
						// Create a reference to the global constant.
						constRefOp := mlir.GoCreateConstantOperation(b.ctx, nil, b.strAttr(symbolName), T, location)
						appendOperation(ctx, constRefOp)
						return resultOf(constRefOp)
					} else {
						// Emit the constant at its point of usage.
						constOp := mlir.GoCreateGlobalConstantOperation(b.ctx, value, nil, location)
						if body != nil {
							mlir.GoGlobalConstantOperationAddBody(constOp, body)
						}
						appendOperation(ctx, constOp)
						return resultOf(constOp)
					}
				},
				T: T,
				b: b,
			})
		}
	}
}

func (b *Builder) emitConstantValue(ctx context.Context, value constant.Value, T types.Type, location mlir.Location) mlir.Value {
	T = types.Unalias(T)
	constT, ok := baseType(T).(*types.Basic)
	if !ok {
		panic("invalid constant type")
	}

	if isUntyped(constT) {
		panic("untyped type is forbidden")
	}

	var resultT mlir.Type
	if typeIs[*types.Named](T) {
		resultT = b.GetStoredType(ctx, T)
	} else {
		resultT = b.GetStoredType(ctx, constT)
	}

	if value == nil {
		constOp := mlir.GoCreateZeroOperation(b.config.Ctx, resultT, location)
		appendOperation(ctx, constOp)
		return resultOf(constOp)
	} else {
		switch constT.Kind() {
		case types.Bool:
			return b.emitConstBool(ctx, constant.BoolVal(value), resultT, location)
		case types.Complex64:
			realValue, _ := constant.Float32Val(constant.Real(value))
			imagValue, _ := constant.Float32Val(constant.Imag(value))
			return b.emitConstComplex64(ctx, realValue, imagValue, resultT, location)
		case types.Complex128:
			realValue, _ := constant.Float64Val(constant.Real(value))
			imagValue, _ := constant.Float64Val(constant.Imag(value))
			return b.emitConstComplex128(ctx, realValue, imagValue, resultT, location)
		case types.Float32:
			constVal, _ := constant.Float32Val(value)
			return b.emitConstFloat32(ctx, constVal, resultT, location)
		case types.Float64:
			constVal, _ := constant.Float64Val(value)
			return b.emitConstFloat64(ctx, constVal, resultT, location)
		case types.Int, types.Int8, types.Int16, types.Int32, types.Int64, types.Uint, types.Uint8, types.Uint16,
			types.Uint32, types.Uint64, types.Uintptr:
			constVal, _ := constant.Int64Val(value)
			return b.emitConstInt(ctx, constVal, resultT, location)
		case types.String:
			return b.emitConstString(ctx, constant.StringVal(value), location)
		case types.UnsafePointer:
			val, _ := constant.Uint64Val(value)
			ptrval := b.emitConstInt(ctx, int64(val), b.uiptr, location)
			op := mlir.GoCreateIntToPtrOperation(b.ctx, ptrval, resultT, location)
			appendOperation(ctx, op)
			return resultOf(op)
		default:
			panic("unhandled constant basic type")
		}
	}
	return nil
}
