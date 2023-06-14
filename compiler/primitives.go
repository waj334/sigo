package compiler

import (
	"context"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createPrimitiveTypes(ctx llvm.LLVMContextRef) {
	typeChannel := llvm.StructCreateNamed(ctx, "channel")
	llvm.StructSetBody(typeChannel, []llvm.LLVMTypeRef{}, false)

	typeInterface := llvm.StructCreateNamed(ctx, "interface")
	llvm.StructSetBody(typeInterface, []llvm.LLVMTypeRef{
		c.ptrType.valueType, // typePtr
		c.ptrType.valueType, // valuePtr
	}, false)

	typeMap := llvm.StructCreateNamed(ctx, "map")
	llvm.StructSetBody(typeMap, []llvm.LLVMTypeRef{}, false)

	typeSlice := llvm.StructCreateNamed(ctx, "slice")
	llvm.StructSetBody(typeSlice, []llvm.LLVMTypeRef{
		c.ptrType.valueType,          // array
		llvm.Int32TypeInContext(ctx), // length
		llvm.Int32TypeInContext(ctx), // capacity
	}, false)

	typeString := llvm.StructCreateNamed(ctx, "string")
	llvm.StructSetBody(typeString, []llvm.LLVMTypeRef{
		c.ptrType.valueType,          // array
		llvm.Int32TypeInContext(ctx), // length
	}, false)
}

func (c *Compiler) int8Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int8TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int16Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int16TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int32Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int32TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) int64Type(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int64TypeInContext(c.currentContext(ctx))
}

func (c *Compiler) boolType(ctx context.Context) llvm.LLVMTypeRef {
	return llvm.Int1TypeInContext(c.currentContext(ctx))
}
