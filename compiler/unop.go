package compiler

import (
	"context"
	"go/token"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createUpOp(ctx context.Context, expr *ssa.UnOp) (value llvm.LLVMValueRef, err error) {
	// Get the value of X
	x, err := c.createExpression(ctx, expr.X)
	if err != nil {
		return nil, err
	}

	// Get the type of X
	xType := c.createType(ctx, expr.Type())

	// Create the respective LLVM operator
	switch expr.Op {
	case token.ARROW:
		// TODO: Channel receive
		panic("Not implemented")
	case token.MUL:
		value = x.UnderlyingValue(ctx)

		if _, isGlobal := expr.X.(*ssa.Global); isGlobal && x.extern {
			value = c.addressOf(ctx, value)
		}

		// Pointer indirection. Create a load operator.
		value = llvm.BuildLoad2(c.builder, xType.valueType, value, "")
	case token.NOT:
		// Logical negation
		value = llvm.BuildNot(c.builder, x.UnderlyingValue(ctx), "")
	case token.SUB:
		// Negation
		value = llvm.BuildNeg(c.builder, x.UnderlyingValue(ctx), "")
	case token.XOR:
		// Bitwise complement
		value = llvm.BuildXor(c.builder, x.UnderlyingValue(ctx), llvm.ConstAllOnes(xType.valueType), "")
	}
	return
}
