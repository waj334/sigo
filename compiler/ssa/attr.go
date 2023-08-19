package ssa

import "omibyte.io/sigo/mlir"

func (b *Builder) intAttr(value int64) mlir.Attribute {
	T := mlir.IntegerTypeGet(b.ctx, 64)
	return mlir.IntegerAttrGet(T, value)
}

func (b *Builder) int32Attr(value int32) mlir.Attribute {
	T := mlir.IntegerTypeGet(b.ctx, 32)
	return mlir.IntegerAttrGet(T, int64(value))
}
