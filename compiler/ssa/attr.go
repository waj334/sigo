package ssa

import (
	"go/types"

	"pkg.si-go.dev/sigo/mlir"
)

func (b *Builder) intAttr(value int64) mlir.Attribute {
	T := mlir.IntegerTypeGet(b.ctx, 64)
	return mlir.IntegerAttrGet(T, value)
}

func (b *Builder) int32Attr(value int32) mlir.Attribute {
	T := mlir.IntegerTypeGet(b.ctx, 32)
	return mlir.IntegerAttrGet(T, int64(value))
}

func (b *Builder) boolAttr(value bool) mlir.Attribute {
	T := mlir.IntegerTypeGet(b.ctx, 1)
	return mlir.IntegerAttrGet(T, func(v bool) int64 {
		if v {
			return 1
		}
		return 0
	}(value))
}

func (b *Builder) strAttr(value string) mlir.Attribute {
	return mlir.StringAttrGet(b.ctx, value)
}

func (b *Builder) strArrayAttr(values ...string) mlir.Attribute {
	strAttrs := make([]mlir.Attribute, len(values))
	for i, v := range values {
		strAttrs[i] = mlir.StringAttrGet(b.ctx, v)
	}
	return mlir.ArrayAttrGet(b.ctx, strAttrs)
}

func (b *Builder) scopeAttr(scope *types.Scope, parent mlir.Attribute) mlir.Attribute {
	return mlir.GoScopeAttrGet(b.ctx, parent, b.unscopedLocation(scope.Pos()), b.unscopedLocation(scope.End()))
}
