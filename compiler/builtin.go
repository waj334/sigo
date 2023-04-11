package compiler

import (
	"context"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createBuiltinCall(ctx context.Context, builtin *ssa.Builtin, args []ssa.Value) (value llvm.LLVMValueRef, err error) {
	// Create the proper builtin call based on the method and the type
	// NOTE: It MUST be defined in the runtime used by this application
	switch builtin.Name() {
	case "append":
	case "cap":
	case "close":
	case "complex":
	case "copy":
	case "delete":
	case "imag":
	case "len":
		value, err = c.createSliceLenCall(ctx, builtin, args[0])
	case "make":
	case "new":
	case "panic":
	case "print":
	case "println":
	case "real":
	case "recover":
	default:
		panic("encountered unknown builtin function")
	}

	return
}

func (c *Compiler) createSliceLenCall(ctx context.Context, builtin *ssa.Builtin, arg ssa.Value) (llvm.LLVMValueRef, error) {
	// Create the value for the argument
	argValue, err := c.createExpression(ctx, arg)
	if err != nil {
		return nil, err
	}

	// The argument should be a struct representing a slice. Create the runtime call with it
	return c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{argValue})
}
