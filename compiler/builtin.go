package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createSliceLenCall(ctx context.Context, builtin *ssa.Builtin, arg ssa.Value) (llvm.LLVMValueRef, error) {
	// Create the value for the argument
	argValue, err := c.createExpression(ctx, arg)
	if err != nil {
		return nil, err
	}

	// The argument should be a struct representing a slice. Create the runtime call with it
	return c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{argValue})
}
