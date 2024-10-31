package ssa

import (
	"context"

	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitTypeConversion(ctx context.Context, X mlir.Value, src types.Type, dest types.Type, location mlir.Location) mlir.Value {
	if typeHasFlags(src, types.IsUntyped) {
		// Attempt to infer the type.
		lhsTypes := currentLhsList(ctx)
		index := currentRhsIndex(ctx)
		if len(lhsTypes) > 0 {
			src = lhsTypes[index]
		} else {
			src = types.Default(src)
		}
	}

	if typeHasFlags(dest, types.IsUntyped) {
		dest = types.Default(dest)
	}

	srcType := b.GetStoredType(ctx, baseType(src))
	destType := b.GetStoredType(ctx, baseType(dest))
	result := X

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
					op := mlir.GoCreateIntTruncateOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				} else if srcWidth == destWidth {
					result = b.bitcastTo(ctx, result, destType, location)
				} else if isUnsigned(destType) {
					op := mlir.GoCreateZeroExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				} else {
					op := mlir.GoCreateSignedExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				}
			case typeHasFlags(dest, types.IsFloat):
				if isSigned(srcType) {
					op := mlir.GoCreateSignedIntToFloatOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				} else {
					op := mlir.GoCreateUnsignedIntToFloatOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				}
			case typeHasFlags(dest, types.IsComplex):
				// TODO: Need operation for this.
				panic("unimplemented")
			case typeHasFlags(dest, types.IsString): // Rune conversion
				// Allocate memory to hold the rune.
				allocOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, srcType, 1, true, location)
				appendOperation(ctx, allocOp)
				storeOp := mlir.GoCreateStoreOperation(b.ctx, X, resultOf(allocOp), location)
				appendOperation(ctx, storeOp)

				// Create and return a string value.
				value := b.emitStringValue(ctx, resultOf(allocOp), b.emitConstInt(ctx, 1, b.si, location), location)

				// Reinterpret as !go.string type.
				result = b.bitcastTo(ctx, value, destType, location)
			case isUnsafePointer(dest):
				op := mlir.GoCreateIntToPtrOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op)
			default:
				panic("unhandled")
			}
		case typeHasFlags(src, types.IsComplex):
			srcWidth := mlir.FloatTypeGetWidth(mlir.ComplexTypeGetElementType(srcType))
			destWidth := mlir.FloatTypeGetWidth(mlir.ComplexTypeGetElementType(destType))
			if destWidth < srcWidth {
				op := mlir.GoCreateComplexTruncateOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op)
			} else {
				op := mlir.GoCreateComplexExtendOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op)
			}
		case typeHasFlags(src, types.IsFloat):
			switch {
			case typeHasFlags(dest, types.IsInteger):
				if isSigned(destType) {
					op := mlir.GoCreateFloatToSignedIntOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				} else {
					op := mlir.GoCreateFloatToUnsignedIntOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				}
			case typeHasFlags(dest, types.IsFloat):
				if mlir.TypeIsAF32(srcType) && mlir.TypeIsAF64(destType) {
					op := mlir.GoCreateFloatExtendOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
				} else {
					op := mlir.GoCreateFloatTruncateOperation(b.ctx, X, destType, location)
					appendOperation(ctx, op)
					result = resultOf(op)
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
				op := mlir.GoCreateRuntimeCallOperation(b.ctx, mangleSymbol("runtime.stringToSlice"), []mlir.Type{b._slice}, []mlir.Value{X}, location)
				appendOperation(ctx, op)

				// Reinterpret as !go.slice type.
				result = b.bitcastTo(ctx, resultOf(op), destType, location)
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
				op := mlir.GoCreateRuntimeCallOperation(b.ctx, mangleSymbol("runtime.sliceToString"), []mlir.Type{b._string}, []mlir.Value{X}, location)
				appendOperation(ctx, op)

				// Reinterpret as !go.string type.
				result = b.bitcastTo(ctx, resultOf(op), destType, location)
			default:
				panic("unhandled")
			}
		case isUnsafePointer(src):
			switch {
			case typeHasFlags(dest, types.IsInteger):
				op := mlir.GoCreatePtrToIntOperation(b.ctx, X, destType, location)
				appendOperation(ctx, op)
				result = resultOf(op)
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
