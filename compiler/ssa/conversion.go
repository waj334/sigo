package ssa

import (
	"context"

	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitTypeConversion(ctx context.Context, X mlir.ValueLike, src types.Type, dest types.Type, location mlir.LocationLike) mlir.Value {
	// Resolve TypeParams to their concrete types in generic function instances.
	if tp, ok := baseType(src).(*types.TypeParam); ok {
		if typeMap := currentTypeMap(ctx); typeMap != nil {
			src = resolveTypeInTypeMap(typeMap[tp.Index()], typeMap)
		}
	}
	if tp, ok := baseType(dest).(*types.TypeParam); ok {
		if typeMap := currentTypeMap(ctx); typeMap != nil {
			dest = resolveTypeInTypeMap(typeMap[tp.Index()], typeMap)
		}
	}

	if typeHasFlags(src, types.IsUntyped) && typeHasFlags(dest, types.IsUntyped) {
		// TODO: Determine the best action to take here.
		return X.AsValue()
	}

	srcType := b.GetStoredType(ctx, baseType(src))
	destType := b.GetStoredType(ctx, baseType(dest))
	result := X.AsValue()

	if !types.Identical(src, dest) {
		if types.Identical(baseType(src), baseType(dest)) {
			// The underlying types are exactly the same, so just perform a bitcast to the destination type.
			destType = b.GetStoredType(ctx, dest)
			return b.bitcastTo(ctx, result, destType, location)
		}

		// Handle interface conversions.
		if types.IsInterface(dest) {
			if types.IsInterface(src) {
				return b.emitChangeType(ctx, dest, X, location)
			}

			// Convert this value to the requested interface type.
			return b.emitInterfaceValue(ctx, dest, src, X, location)
		}

		switch {
		case typeHasFlags(src, types.IsInteger):
			switch {
			case typeHasFlags(dest, types.IsInteger):
				srcWidth := b.widthOf(srcType)
				destWidth := b.widthOf(destType)
				if srcWidth > destWidth {
					op := goir.NewIntTruncateOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				} else if srcWidth == destWidth {
					result = b.bitcastTo(ctx, result, destType, location)
				} else if isUnsigned(srcType) {
					op := goir.NewZeroExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				} else {
					op := goir.NewSignedExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				}
			case typeHasFlags(dest, types.IsFloat):
				if isSigned(srcType) {
					op := goir.NewSignedIntToFloatOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				} else {
					op := goir.NewUnsignedIntToFloatOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				}
			case typeHasFlags(dest, types.IsComplex):
				// TODO: Need operation for this.
				panic("unimplemented")
			case typeHasFlags(dest, types.IsString): // Rune conversion
				// Allocate memory to hold the rune.
				allocOp := goir.NewAllocaOperation(b.ctx, b.ptr, srcType, 1, true, location)
				appendOperation(ctx, allocOp)
				b.emitStore(ctx, X, resultOf(allocOp), location)

				// Create and return a string value.
				value := b.emitStringValue(ctx, resultOf(allocOp), b.emitConstInt(ctx, 1, b.si, location), location)

				// Reinterpret as !go.string type.
				result = b.bitcastTo(ctx, value, destType, location)
			case isUnsafePointer(dest):
				op := goir.NewIntToPtrOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			default:
				panic("unhandled")
			}
		case typeHasFlags(src, types.IsComplex):
			srcComplexType, _ := mlir.AsComplexType(srcType)
			destComplexType, _ := mlir.AsComplexType(destType)
			srcWidth := b.widthOf(srcComplexType.ElementType())
			destWidth := b.widthOf(destComplexType.ElementType())
			if destWidth < srcWidth {
				op := goir.NewComplexTruncateOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			} else {
				op := goir.NewComplexExtendOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			}
		case typeHasFlags(src, types.IsFloat):
			switch {
			case typeHasFlags(dest, types.IsInteger):
				if isSigned(destType) {
					op := goir.NewFloatToSignedIntOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				} else {
					op := goir.NewFloatToUnsignedIntOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				}
			case typeHasFlags(dest, types.IsFloat):
				srcWidth := b.widthOf(srcType)
				destWidth := b.widthOf(destType)
				if srcWidth < destWidth {
					op := goir.NewFloatExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				} else {
					op := goir.NewFloatTruncateOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op).AsValue()
				}
			case typeHasFlags(dest, types.IsComplex):
				// TODO: Need operation for this.
				panic("unimplemented")
			default:
				panic("unhandled")
			}
		case typeHasFlags(src, types.IsString):
			switch {
			case typeIs[*types.Slice](dest):
				op := goir.NewStringToSliceOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			case typeHasFlags(dest, types.IsString):
				// NOTE: The input is probably untyped. Just perform a bitcast.
				op := goir.NewBitcastOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			default:
				panic("unhandled")
			}
		case typeIs[*types.Pointer](src):
			switch {
			case isUnsafePointer(dest), typeIs[*types.Pointer](src):
				result = b.bitcastTo(ctx, X, destType, location)
			default:
				panic("unhandled")
			}
		case typeIs[*types.Slice](src):
			switch {
			case typeHasFlags(dest, types.IsString):
				op := goir.NewSliceToStringOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			default:
				panic("unhandled")
			}
		case isUnsafePointer(src):
			switch {
			case typeHasFlags(dest, types.IsInteger):
				op := goir.NewPtrToIntOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op).AsValue()
			case typeIs[*types.Pointer](dest):
				result = b.bitcastTo(ctx, X, destType, location)
			default:
				panic("unhandled")
			}
		default:
			panic("unhandled")
		}
	}

	// Perform a final bitcast if the destination type is named.
	if typeIs[*types.Named](dest) && !types.Identical(src, dest) {
		T := b.GetStoredType(ctx, dest)
		result = b.bitcastTo(ctx, result, T, location)
	}

	return result
}
