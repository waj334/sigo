package compiler

import (
	"context"
	"fmt"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createBinOp(ctx context.Context, expr *ssa.BinOp) (value llvm.LLVMValueRef) {
	// Get operand values
	x := c.createExpression(ctx, expr.X).UnderlyingValue(ctx)
	y := c.createExpression(ctx, expr.Y).UnderlyingValue(ctx)

	if basicType, ok := expr.Y.Type().Underlying().(*types.Basic); ok {
		//if basicType.Info()&types.IsUntyped != 0 && basicType.Info()&types.IsInteger != 0 {
		if basicType.Info()&types.IsInteger != 0 {
			// Cast the untyped integer to that of the lhs
			y = llvm.BuildIntCast2(c.builder, y, llvm.TypeOf(x), basicType.Info()&types.IsUnsigned == 0, "untyped_int_cast")
		}
	}

	if llvm.IsAFunction(x) != nil && llvm.IsAFunction(y) == nil {
		panic("mismatched function types")
	} else if !llvm.TypeIsEqual(llvm.TypeOf(x), llvm.TypeOf(y)) {
		panic(fmt.Sprintf("operand types do not match: %s != %s",
			llvm.PrintTypeToString(llvm.TypeOf(x)),
			llvm.PrintTypeToString(llvm.TypeOf(y))),
		)
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
		} else if typeInfo&types.IsString != 0 {
			// Concatenate the strings using the respective runtime call
			var stringValue llvm.LLVMValueRef
			stringValue = c.createRuntimeCall(ctx, "stringConcat", []llvm.LLVMValueRef{
				c.addressOf(ctx, x),
				c.addressOf(ctx, y),
			})

			// Load the resulting string values
			value = llvm.BuildLoad2(c.builder, llvm.GetTypeByName2(c.currentContext(ctx), "string"), c.addressOf(ctx, stringValue), "")
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
				value = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x), c.addressOf(ctx, y)})
			} else {
				// Comparing an interface against anything a non-interface. EQL = false in this scenario
				value = llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), 0, false)
			}
		} else if typeInfo&types.IsString != 0 {
			value = c.createRuntimeCall(ctx, "stringCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x), c.addressOf(ctx, y)})
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealOEQ, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntEQ, x, y, "")
		}
	case token.NEQ:
		if types.IsInterface(expr.X.Type()) || types.IsInterface(expr.Y.Type()) {
			if types.IsInterface(expr.X.Type()) && types.IsInterface(expr.X.Type()) {
				// runtime call for interface equality
				value = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x), c.addressOf(ctx, y)})

				// Invert the result
				value = llvm.BuildNot(c.builder, value, "")
			} else {
				// Comparing an interface against anything a non-interface. NEQ = true in this scenario
				value = llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), 1, false)
			}
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealONE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntNE, x, y, "")
		}
	case token.LSS:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealOLT, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.IntULT, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntSLT, x, y, "")
		}
	case token.LEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealOLE, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.IntULE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntSLE, x, y, "")
		}
	case token.GTR:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealOGT, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.IntUGT, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntSGT, x, y, "")
		}
	case token.GEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.RealOGE, x, y, "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.IntUGE, x, y, "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.IntSGE, x, y, "")
		}
	}

	return
}
