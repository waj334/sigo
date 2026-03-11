package ssa

import (
	"context"
	"go/ast"
	"go/constant"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitConstantDecl(ctx context.Context, decl *ast.GenDecl) {
	for _, spec := range decl.Specs {
		valueSpec := spec.(*ast.ValueSpec)
		for i, ident := range valueSpec.Names {
			location := b.location(ctx, ident.Pos())
			obj := b.objectOf(ctx, ident).(*types.Const)
			symbolName := qualifiedName(obj.Name(), obj.Pkg())
			constT := b.typeOf(ctx, ident)
			T := b.GetStoredType(ctx, constT)

			fromObj := func(obj *types.Const) mlir.AttributeLike {
				litType := types.Unalias(constT).Underlying().(*types.Basic)
				switch litType.Kind() {
				case types.Bool, types.UntypedBool:
					return b.boolAttr(constant.BoolVal(obj.Val()))
				case types.Complex64:
					realValue, _ := constant.Float32Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float32Val(constant.Imag(obj.Val()))
					return goir.NewComplexAttr(b.ctx, b.f32, float64(realValue), float64(imagValue))
				case types.Complex128, types.UntypedComplex:
					realValue, _ := constant.Float64Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float64Val(constant.Imag(obj.Val()))
					return goir.NewComplexAttr(b.ctx, b.f64, realValue, imagValue)
				case types.Float32:
					k, _ := constant.Float32Val(obj.Val())
					return mlir.NewFloatAttr(b.ctx, b.f32, float64(k))
				case types.Float64, types.UntypedFloat:
					k, _ := constant.Float64Val(obj.Val())
					return mlir.NewFloatAttr(b.ctx, b.f32, k)
				case types.Int, types.Int8, types.Int16, types.Int32, types.Int64, types.UntypedInt, types.UntypedRune:
					constVal, _ := constant.Int64Val(obj.Val())
					return b.intAttr(constVal)
				case types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.Uintptr:
					constVal, _ := constant.Uint64Val(obj.Val())
					return b.intAttr(int64(constVal))
				case types.String, types.UntypedString:
					return b.strAttr(constant.StringVal(obj.Val()))
				default:
					panic("unreachable: " + litType.String())
				}
			}

			// Is this constant at the global scope?
			if obj.Parent() == obj.Pkg().Scope() {
				// Generate the constant body.
				body := mlir.NewBlock(nil, nil)
				{ // Start constant body block.
					ctx := newContextWithCurrentBlock(ctx)
					setCurrentBlock(ctx, body)

					var result mlir.Value
					if valueSpec.Values != nil {
						result = b.emitExpr(ctx, valueSpec.Values[i])[0].AsValue()
					} else {
						value := fromObj(obj)
						constOp := goir.NewConstantOperation(b.ctx, value, nil, T, location)
						appendOperation(ctx, constOp)
						result = resultOf(constOp).AsValue()
					}

					// Create the yield operation.
					yieldOp := goir.NewYieldOperation(b.ctx, result, location)
					appendOperation(ctx, yieldOp)
				} // End of constant body block.

				// Emit a global constant that will be referred to later when used.
				// NOTE: value is nil if the constant value was NOT a literal value.
				constOp := goir.NewGlobalConstantOperation(b.ctx, nil, b.strAttr(symbolName), location)
				goir.GlobalConstantOperationAddBody(constOp, body)

				appendOperation(ctx, constOp)

				b.setAddr(ctx, ident, ConstantValue{
					Emitter: func(ctx context.Context, location mlir.LocationLike) mlir.Value {
						// Create a reference to the global constant.
						constRefOp := goir.NewConstantOperation(b.ctx, nil, b.strAttr(symbolName), T, location)
						appendOperation(ctx, constRefOp)
						return resultOf(constRefOp).AsValue()
					},
					T: T,
					b: b,
				})
			} else {
				value := fromObj(obj)
				b.setAddr(ctx, ident, ConstantValue{
					Emitter: func(ctx context.Context, location mlir.LocationLike) mlir.Value {
						// Create a local constant.
						constRefOp := goir.NewConstantOperation(b.ctx, value, nil, T, location)
						appendOperation(ctx, constRefOp)
						return resultOf(constRefOp).AsValue()
					},
					T: T,
					b: b,
				})
			}
		}
	}
}

func (b *Builder) emitConstantValue(ctx context.Context, value constant.Value, T types.Type, location mlir.LocationLike) mlir.Value {
	T = types.Unalias(T)
	constT, ok := baseType(T).(*types.Basic)
	if !ok {
		panic("invalid constant type")
	}

	var resultT mlir.TypeLike
	if typeIs[*types.Named](T) {
		resultT = b.GetStoredType(ctx, T)
	} else {
		resultT = b.GetStoredType(ctx, constT)
	}

	if value == nil {
		constOp := goir.NewZeroOperation(b.config.Ctx, resultT, location)
		appendOperation(ctx, constOp)
		return resultOf(constOp).AsValue()
	}

	switch constT.Kind() {
	case types.Bool, types.UntypedBool:
		return b.emitConstBool(ctx, constant.BoolVal(value), resultT, location)
	case types.Complex64:
		realValue, _ := constant.Float32Val(constant.Real(value))
		imagValue, _ := constant.Float32Val(constant.Imag(value))
		return b.emitConstComplex64(ctx, realValue, imagValue, resultT, location)
	case types.Complex128, types.UntypedComplex:
		realValue, _ := constant.Float64Val(constant.Real(value))
		imagValue, _ := constant.Float64Val(constant.Imag(value))
		return b.emitConstComplex128(ctx, realValue, imagValue, resultT, location)
	case types.Float32:
		constVal, _ := constant.Float32Val(value)
		return b.emitConstFloat32(ctx, constVal, resultT, location)
	case types.Float64, types.UntypedFloat:
		constVal, _ := constant.Float64Val(value)
		return b.emitConstFloat64(ctx, constVal, resultT, location)
	case types.Int, types.Int8, types.Int16, types.Int32, types.Int64, types.Uint, types.Uint8, types.Uint16,
		types.Uint32, types.Uint64, types.Uintptr, types.UntypedInt, types.UntypedRune:
		constVal, _ := constant.Int64Val(value)
		return b.emitConstInt(ctx, constVal, resultT, location)
	case types.String, types.UntypedString:
		return b.emitConstString(ctx, constant.StringVal(value), resultT, location)
	case types.UnsafePointer:
		val, _ := constant.Uint64Val(value)
		ptrval := b.emitConstInt(ctx, int64(val), b.uiptr, location)
		op := goir.NewIntToPtrOperation(b.ctx, ptrval, resultT, location)
		appendOperation(ctx, op)
		return resultOf(op).AsValue()
	default:
		panic("unhandled constant basic type")
	}
}
