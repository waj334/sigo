package ssa

import (
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) intAttr(value int64) mlir.IntegerAttr {
	T := mlir.NewIntegerType(b.ctx, 64)
	return mlir.NewIntegerAttr(T, value)
}

func (b *Builder) int32Attr(value int32) mlir.IntegerAttr {
	T := mlir.NewIntegerType(b.ctx, 32)
	return mlir.NewIntegerAttr(T, int64(value))
}

func (b *Builder) boolAttr(value bool) mlir.BoolAttr {
	return mlir.NewBoolAttr(b.ctx, value)
}

func (b *Builder) strAttr(value string) mlir.StringAttr {
	return mlir.NewStringAttr(b.ctx, value)
}

func (b *Builder) strArrayAttr(values ...string) mlir.ArrayAttr {
	strAttrs := make([]mlir.StringAttr, len(values))
	for i, v := range values {
		strAttrs[i] = mlir.NewStringAttr(b.ctx, v)
	}
	return mlir.NewArrayAttr(b.ctx, strAttrs)
}

func (b *Builder) scopeAttr(scope *types.Scope, parent goir.ScopeAttr) goir.ScopeAttr {
	return goir.NewScopeAttr(b.ctx, parent, b.unscopedLocation(scope.Pos()), b.unscopedLocation(scope.End()))
}
