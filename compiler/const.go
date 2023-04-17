package compiler

import (
	"context"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createConstValue(ctx context.Context, valueType llvm.LLVMTypeRef) (value llvm.LLVMValueRef) {
	switch llvm.GetTypeKind(valueType) {
	case llvm.IntegerTypeKind:
		value = llvm.ConstInt(valueType, 0, false)
	case llvm.FloatTypeKind, llvm.DoubleTypeKind:
		value = llvm.ConstReal(valueType, 0)
	case llvm.StructTypeKind:
		count := int(llvm.CountStructElementTypes(valueType))
		var values []llvm.LLVMValueRef
		for i := 0; i < count; i++ {
			elementType := llvm.StructGetTypeAtIndex(valueType, uint(i))
			values = append(values, c.createConstValue(ctx, elementType))
		}
		value = llvm.ConstNamedStruct(valueType, values)
	case llvm.PointerTypeKind:
		value = llvm.ConstPointerNull(valueType)
	case llvm.FunctionTypeKind:
		// TODO: Still need to determine how to handle anonymous functions
		panic("not implemented")
	default:
		panic("cannot create const value of specified type")
	}
	return
}

func (c *Compiler) createRuntimeConst(ctx context.Context, typename string) (value llvm.LLVMValueRef) {
	runtimeType := llvm.GetTypeByName2(c.currentContext(ctx), typename)
	if runtimeType != nil {
		value = c.createConstValue(ctx, runtimeType)
	}
	return
}
