package ssa

import (
	"context"

	"go/ast"
	"go/constant"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitConstantDecl(ctx context.Context, decl *ast.GenDecl) {
	for _, spec := range decl.Specs {
		spec := spec.(*ast.ValueSpec)
		for _, ident := range spec.Names {
			obj := b.objectOf(ctx, ident).(*types.Const)
			constT := b.typeOf(ctx, ident)

			if typeHasFlags(constT, types.IsUntyped) {
				constT = types.Default(constT)
			}

			T := b.GetStoredType(ctx, constT)
			switch objType := baseType(constT).(type) {
			case *types.Basic:
				switch objType.Kind() {
				case types.Bool:
					val := constant.BoolVal(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstBool(ctx, val, location)
						},
						T: T,
						b: b,
					})
				case types.Complex64:
					realValue, _ := constant.Float32Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float32Val(constant.Imag(obj.Val()))
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstComplex64(ctx, realValue, imagValue, b.c64, location)
						},
						T: T,
						b: b,
					})
				case types.Complex128:
					realValue, _ := constant.Float64Val(constant.Real(obj.Val()))
					imagValue, _ := constant.Float64Val(constant.Imag(obj.Val()))
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstComplex128(ctx, realValue, imagValue, b.c128, location)
						},
						T: T,
						b: b,
					})
				case types.Float32:
					val, _ := constant.Float32Val(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstFloat32(ctx, val, location)
						},
						T: T,
						b: b,
					})
				case types.Float64:
					val, _ := constant.Float64Val(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstFloat64(ctx, val, location)
						},
						T: T,
						b: b,
					})
				case types.Int, types.Int8, types.Int16, types.Int32, types.Int64:
					val, _ := constant.Int64Val(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstInt(ctx, val, T, location)
						},
						T: T,
						b: b,
					})
				case types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.Uintptr:
					val, _ := constant.Uint64Val(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstInt(ctx, int64(val), T, location)
						},
						T: T,
						b: b,
					})
				case types.String:
					val := constant.StringVal(obj.Val())
					b.setAddr(ctx, ident, ConstantValue{
						Emitter: func(ctx context.Context, location mlir.Location) mlir.Value {
							return b.emitConstString(ctx, val, location)
						},
						T: T,
						b: b,
					})
				default:
					panic("invalid constant basic type")
				}
			default:
				panic("unhandled constant type")
			}
		}
	}
}

func (b *Builder) emitConstantValue(ctx context.Context, value constant.Value, T types.Type, location mlir.Location) mlir.Value {
	constT, ok := baseType(T).(*types.Basic)
	if !ok {
		panic("invalid constant type")
	}

	if isUntyped(constT) {
		panic("untyped type is forbidden")
		//constT = types.Default(constT).(*types.Basic)
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
			return b.emitConstBool(ctx, constant.BoolVal(value), location)
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
			return b.emitConstFloat32(ctx, constVal, location)
		case types.Float64:
			constVal, _ := constant.Float64Val(value)
			return b.emitConstFloat64(ctx, constVal, location)
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
