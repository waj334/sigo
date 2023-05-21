package compiler

import (
	"context"
	"fmt"
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

	if basicType, ok := expr.Y.Type().Underlying().(*types.Basic); ok {
		//if basicType.Info()&types.IsUntyped != 0 && basicType.Info()&types.IsInteger != 0 {
		if basicType.Info()&types.IsInteger != 0 {
			// Cast the untyped integer to that of the lhs
			y.LLVMValueRef = llvm.BuildIntCast2(c.builder, y.UnderlyingValue(ctx), llvm.TypeOf(x), basicType.Info()&types.IsUnsigned == 0, "untyped_int_cast")
		}
	}

	if llvm.IsAFunction(x.UnderlyingValue(ctx)) != nil && llvm.IsAFunction(y.UnderlyingValue(ctx)) == nil {
		panic("mismatched function types")
	} else if !llvm.TypeIsEqual(llvm.TypeOf(x.UnderlyingValue(ctx)), llvm.TypeOf(y.UnderlyingValue(ctx))) {
		panic(fmt.Sprintf("operand types do not match: %s != %s",
			llvm.PrintTypeToString(llvm.TypeOf(x.UnderlyingValue(ctx))),
			llvm.PrintTypeToString(llvm.TypeOf(y.UnderlyingValue(ctx)))),
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
			value = llvm.BuildFAdd(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsString != 0 {
			// Concatenate the strings using the respective runtime call
			var stringValue llvm.LLVMValueRef
			stringValue, err = c.createRuntimeCall(ctx, "stringConcat", []llvm.LLVMValueRef{
				c.addressOf(ctx, x.UnderlyingValue(ctx)),
				c.addressOf(ctx, y.UnderlyingValue(ctx)),
			})
			if err != nil {
				return nil, err
			}

			// Load the resulting string values
			value = llvm.BuildLoad2(c.builder, llvm.GetTypeByName2(c.currentContext(ctx), "string"), c.addressOf(ctx, stringValue), "")
		} else {
			value = llvm.BuildAdd(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.SUB:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFSub(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildSub(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.MUL:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFMul(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildMul(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.QUO:
		// Choose the correct division operator based on the lhs value
		// type. Both operands are expected to be of the same
		//signedness as per the Go language spec.
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildUDiv(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFDiv(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildSDiv(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.REM:
		// Choose the correct remainder operator based on the lhs value
		// type. Both operands are expected to be of the same
		//signedness as per the Go language spec.
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildURem(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFRem(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildSRem(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.AND:
		value = llvm.BuildAnd(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
	case token.OR:
		value = llvm.BuildOr(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
	case token.XOR:
		value = llvm.BuildXor(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
	case token.SHL:
		value = llvm.BuildShl(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
	case token.SHR:
		if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildLShr(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildAShr(c.builder, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.AND_NOT:
		negY := llvm.BuildNeg(c.builder, y.UnderlyingValue(ctx), "and_not_neg_y")
		value = llvm.BuildAnd(c.builder, x.UnderlyingValue(ctx), negY, "")
	case token.EQL:
		if types.IsInterface(expr.X.Type()) || types.IsInterface(expr.Y.Type()) {
			if types.IsInterface(expr.X.Type()) && types.IsInterface(expr.X.Type()) {
				// runtime call for interface equality
				value, err = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x.UnderlyingValue(ctx)), c.addressOf(ctx, y.UnderlyingValue(ctx))})
				if err != nil {
					return nil, err
				}
			} else {
				// Comparing an interface against anything a non-interface. EQL = false in this scenario
				value = llvm.ConstInt(llvm.Int1TypeInContext(c.currentContext(ctx)), 0, false)
			}
		} else if typeInfo&types.IsString != 0 {
			value, err = c.createRuntimeCall(ctx, "stringCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x.UnderlyingValue(ctx)), c.addressOf(ctx, y.UnderlyingValue(ctx))})
			if err != nil {
				return nil, err
			}
		} else if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOEQ, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntEQ, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.NEQ:
		if types.IsInterface(expr.X.Type()) || types.IsInterface(expr.Y.Type()) {
			if types.IsInterface(expr.X.Type()) && types.IsInterface(expr.X.Type()) {
				// runtime call for interface equality
				value, err = c.createRuntimeCall(ctx, "interfaceCompare", []llvm.LLVMValueRef{c.addressOf(ctx, x.UnderlyingValue(ctx)), c.addressOf(ctx, y.UnderlyingValue(ctx))})
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
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealONE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntNE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.LSS:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOLT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntULT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSLT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.LEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOLE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntULE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSLE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.GTR:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOGT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntUGT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSGT, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	case token.GEQ:
		if typeInfo&types.IsFloat != 0 {
			value = llvm.BuildFCmp(c.builder, llvm.LLVMRealOGE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else if typeInfo&types.IsUnsigned != 0 {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntUGE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		} else {
			value = llvm.BuildICmp(c.builder, llvm.LLVMIntSGE, x.UnderlyingValue(ctx), y.UnderlyingValue(ctx), "")
		}
	}

	return
}
