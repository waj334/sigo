package compiler

import (
	"context"
	"go/token"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createUnOp(ctx context.Context, expr *ssa.UnOp) (value llvm.LLVMValueRef) {
	// Get the value of X
	x := c.createExpression(ctx, expr.X)
	xValue := x.UnderlyingValue(ctx)

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
		value = llvm.BuildNot(c.builder, xValue, "")
	case token.SUB:
		// Negation
		value = llvm.BuildNeg(c.builder, xValue, "")
	case token.XOR:
		// Bitwise complement
		value = llvm.BuildXor(c.builder, xValue, llvm.ConstAllOnes(xType.valueType), "")
	}
	return
}
