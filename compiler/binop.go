package compiler

import (
	"context"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createBinOp(ctx context.Context, expr *ssa.BinOp) (value llvm.LLVMValueRef, err error) {
	// Get operand values
	x, err := c.createExpression(ctx, expr.X)
	if err != nil {
		return nil, err
	}

	y, err := c.createExpression(ctx, expr.Y)
	if err != nil {
		return nil, err
	}

	if !llvm.TypeIsEqual(llvm.TypeOf(x), llvm.TypeOf(y)) {
		panic("operand types do not match")
	}

	var typeInfo types.BasicInfo
	if basic, ok := expr.X.Type().(*types.Basic); ok {
		typeInfo = basic.Info()
	}
	// Create the respective operator
	switch expr.Op {
	case token.ADD:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFAdd(c.builder, x, y, "")
		} else {
			value = llvm.BuildAdd(c.builder, x, y, "")
		}
	case token.SUB:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFSub(c.builder, x, y, "")
		} else {
			value = llvm.BuildSub(c.builder, x, y, "")
		}
	case token.MUL:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFMul(c.builder, x, y, "")
		} else {
			value = llvm.BuildMul(c.builder, x, y, "")
		}
	case token.QUO:
		// Choose the correct division operator based on the lhs value
		// type. Both operands are expected to be of the same
		//signedness as per the Go language spec.
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildUDiv(c.builder, x, y, "")
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFDiv(c.builder, x, y, "")
		} else {
			value = llvm.BuildSDiv(c.builder, x, y, "")
		}
	case token.REM:
		// Choose the correct remainder operator based on the lhs value
		// type. Both operands are expected to be of the same
		//signedness as per the Go language spec.
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildURem(c.builder, x, y, "")
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFRem(c.builder, x, y, "")
		} else {
			value = llvm.BuildSRem(c.builder, x, y, "")
		}
	case token.AND:
		value = llvm.BuildAnd(c.builder, x, y, "")
	case token.OR:
		value = llvm.BuildOr(c.builder, x, y, "")
	case token.XOR:
		value = llvm.BuildXor(c.builder, x, y, "")
	case token.SHL:
		value = llvm.BuildShl(c.builder, x, y, "")
	case token.SHR:
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildLShr(c.builder, x, y, "")
		} else {
			value = llvm.BuildAShr(c.builder, x, y, "")
		}
	case token.AND_NOT:
		negY := llvm.BuildNeg(c.builder, y, "and_not_neg_y")
		value = llvm.BuildAnd(c.builder, x, negY, "")
	case token.EQL:
		if types.IsInterface(expr.X.Type()) || types.IsInterface(expr.Y.Type()) {
			if types.IsInterface(expr.X.Type()) && types.IsInterface(expr.X.Type()) {
				// runtime call for interface equality
				value, err = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(x), c.addressOf(y)})
				if err != nil {
					return nil, err
				}
			} else {
				// Comparing an interface against anything a non-interface. EQL = false in this scenario
				value = llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), 0, false)
			}
		} else if typeInfo&types.IsString != 0 {
			value, err = c.createRuntimeCall(ctx, "stringCompare", []llvm.LLVMValueRef{c.addressOf(x), c.addressOf(y)})
			if err != nil {
				return nil, err
			}
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOEQ, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntEQ, x, y, "")
		}
	case token.NEQ:
		if types.IsInterface(expr.X.Type()) || types.IsInterface(expr.Y.Type()) {
			if types.IsInterface(expr.X.Type()) && types.IsInterface(expr.X.Type()) {
				// runtime call for interface equality
				value, err = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(x), c.addressOf(y)})
				if err != nil {
					return nil, err
				}
				// Invert the result
				value = llvm.BuildNot(c.builder, value, "")
			} else {
				// Comparing an interface against anything a non-interface. NEQ = true in this scenario
				value = llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), 1, false)
			}
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealONE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntNE, x, y, "")
		}
	case token.LSS:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOLT, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntULT, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSLT, x, y, "")
		}
	case token.LEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOLE, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntULE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSLE, x, y, "")
		}
	case token.GTR:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOGT, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntUGT, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSGT, x, y, "")
		}
	case token.GEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOGE, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntUGE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSGE, x, y, "")
		}
	}

	return
}
