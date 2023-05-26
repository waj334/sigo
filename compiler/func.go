package compiler

import (
	"context"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createClosure(ctx context.Context, signature *types.Signature, callArgs []ssa.Value) (llvm.LLVMValueRef, llvm.LLVMValueRef) {
	// Create a struct type to represent the parameters
	var paramTypes []llvm.LLVMTypeRef
	for _, callArg := range callArgs {
		//param := signature.Params().At(i)
		//paramTypes = append(paramTypes, c.createType(ctx, param.Type().Underlying()).valueType)
		callArgType := c.createType(ctx, callArg.Type().Underlying())
		paramTypes = append(paramTypes, callArgType.valueType)
	}

	// TODO: Handle return type

	// Create the struct holding the parameters
	callArgValues, _ := c.createValues(ctx, callArgs)
	paramsStructType := llvm.StructTypeInContext(c.currentContext(ctx), paramTypes, false)
	paramStructValue := llvm.BuildAlloca(c.builder, paramsStructType, "closure_params")
	for i, callArg := range callArgValues.Ref(ctx) {
		//callArgType := c.createType(ctx, callArgs[i].Type())
		argAddr := llvm.BuildStructGEP2(c.builder, paramsStructType, paramStructValue, uint(i), "closure_param_arg_addr")
		llvm.BuildStore(c.builder, callArg, argAddr)
	}

	closureFnType := llvm.FunctionType(
		llvm.VoidTypeInContext(c.currentContext(ctx)),
		[]llvm.LLVMTypeRef{llvm.PointerType(paramsStructType, 0)},
		false)
	closureFnValue := llvm.AddFunction(c.module, "closure", closureFnType)

	currentBlock := llvm.GetInsertBlock(c.builder)
	defer llvm.PositionBuilderAtEnd(c.builder, currentBlock)

	block := llvm.AppendBasicBlockInContext(c.currentContext(ctx), closureFnValue, "closure_entry")
	llvm.PositionBuilderAtEnd(c.builder, block)
	params := llvm.GetParam(closureFnValue, 0)

	var args []llvm.LLVMValueRef
	for i, _ := range callArgValues {
		paramType := c.createType(ctx, callArgs[i].Type())
		argAddr := llvm.BuildStructGEP2(c.builder, paramsStructType, params, uint(i), "closure_arg_addr")
		arg := llvm.BuildLoad2(c.builder, paramType.valueType, argAddr, "closure_arg_load")
		args = append(args, arg)
	}

	// Get the actual function to call
	actualFn := c.functions[signature]
	//actualFnType := c.createType(ctx, signature)
	//actualFnType := c.createType(ctx, signature)

	llvm.BuildCall2(c.builder, actualFn.llvmType, actualFn.value, args, "")

	// TODO: Handle return values
	llvm.BuildRetVoid(c.builder)

	return closureFnValue, paramStructValue
}
