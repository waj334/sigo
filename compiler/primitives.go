package compiler

import (
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
