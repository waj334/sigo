package compiler

import (
	"context"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"hash/fnv"
	"io"

	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createFunctionType(ctx context.Context, signature *types.Signature, hasFreeVars bool) llvm.LLVMTypeRef {
	var returnValueTypes []llvm.LLVMTypeRef
	var argValueTypes []llvm.LLVMTypeRef
	var returnType llvm.LLVMTypeRef

	if fnType, ok := c.signatures[signature]; ok {
		return fnType
	}

	if numArgs := signature.Results().Len(); numArgs == 0 {
		returnType = llvm.VoidTypeInContext(c.currentContext(ctx))
	} else if numArgs == 1 {
		returnType = c.createType(ctx, signature.Results().At(0).Type()).valueType
	} else {
		// Create a struct type to store the return values into
		for i := 0; i < numArgs; i++ {
			resultType := signature.Results().At(i).Type()
			returnValueTypes = append(returnValueTypes, c.createType(ctx, resultType).valueType)
		}
		returnType = llvm.StructTypeInContext(c.currentContext(ctx), returnValueTypes, false)
	}

	// Create the receiver type if this function is a method
	if signature.Recv() != nil {
		var receiverType llvm.LLVMTypeRef
		if _, isInterface := signature.Recv().Type().Underlying().(*types.Interface); isInterface {
			receiverType = c.ptrType.valueType
		} else {
			receiverType = c.createType(ctx, signature.Recv().Type()).valueType
		}
		argValueTypes = append(argValueTypes, receiverType)
	}

	// Create types for the arguments
	for i := 0; i < signature.Params().Len(); i++ {
		arg := signature.Params().At(i)
		argType := c.createType(ctx, arg.Type())
		argValueTypes = append(argValueTypes, argType.valueType)
	}

	// Free vars will be loaded from a pointer representing the state
	if hasFreeVars {
		argValueTypes = append(argValueTypes, c.ptrType.valueType)
	}

	// Create the function type
	return llvm.FunctionType(returnType, argValueTypes, signature.Variadic())
}

func (c *Compiler) createClosure(ctx context.Context, fn *ssa.Function, closureCtxType llvm.LLVMTypeRef) llvm.LLVMValueRef {
	// Get the actual function to call
	actualFn := c.functions[fn]

	// Get the context struct type that will hold the parameters forwarded by the closure
	//closureCtxType := c.createClosureContextType(ctx, fn)

	// Get the return type of the function to be called by this closure
	returnType := llvm.GetReturnType(c.signatures[fn.Signature])
	closureFnType := llvm.FunctionType(
		returnType,
		[]llvm.LLVMTypeRef{llvm.PointerType(closureCtxType, 0)},
		false)
	linkname := c.symbolName(c.currentPackage(ctx).Pkg, "closure")
	closureFnValue := llvm.AddFunction(c.module, linkname, closureFnType)

	currentBlock := llvm.GetInsertBlock(c.builder)
	defer llvm.PositionBuilderAtEnd(c.builder, currentBlock)

	block := llvm.AppendBasicBlockInContext(c.currentContext(ctx), closureFnValue, "closure_entry")
	llvm.PositionBuilderAtEnd(c.builder, block)
	params := llvm.GetParam(closureFnValue, 0)

	var args []llvm.LLVMValueRef
	for i, t := range llvm.GetStructElementTypes(closureCtxType) {
		argAddr := llvm.BuildStructGEP2(c.builder, closureCtxType, params, uint(i), "closure_arg_addr")
		arg := llvm.BuildLoad2(c.builder, t, argAddr, "closure_arg_load")
		args = append(args, arg)
	}

	// Build the call
	ret := llvm.BuildCall2(c.builder, actualFn.llvmType, actualFn.value, args, "")
	if returnType == llvm.VoidTypeInContext(c.currentContext(ctx)) {
		llvm.BuildRetVoid(c.builder)
	} else {
		llvm.BuildRet(c.builder, ret)
	}

	return closureFnValue
}

func (c *Compiler) createClosureContextType(ctx context.Context, fn *ssa.Function) llvm.LLVMTypeRef {
	/*var paramTypes []llvm.LLVMTypeRef

	signature := fn.Signature

	// Get the function's parameter types
	for i := 0; i < signature.Params().Len(); i++ {
		t := c.createType(ctx, signature.Params().At(i).Type()).valueType
		paramTypes = append(paramTypes, t)
	}

	// Bound values are appended to the closure's arguments. Add those to the context
	for _, bound := range fn.FreeVars {
		t := c.createType(ctx, bound.Type()).valueType
		paramTypes = append(paramTypes, t)
	}*/

	paramTypes := llvm.GetParamTypes(c.signatures[fn.Signature])
	paramsStructType := llvm.StructTypeInContext(c.currentContext(ctx), paramTypes, false)
	return paramsStructType
}

func (c *Compiler) createClosureContext(ctx context.Context, closureCtxType llvm.LLVMTypeRef, args []llvm.LLVMValueRef) llvm.LLVMValueRef {
	paramTypes := llvm.GetStructElementTypes(closureCtxType)

	// Fail early if the number arguments doesn't match the number function parameters
	if len(args) != len(paramTypes) {
		panic("len(args) != len(params)")
	}

	// Create the function argument values
	//argValues := c.createValues(ctx, args)
	//bound := c.createValues(ctx, bindings)

	// Create an instance of the params struct
	x := c.createAlloca(ctx, closureCtxType, "closure.args")
	for i, arg := range args /*append(argValues, bound...)*/ {
		// Verify argument types against param types
		if !llvm.TypeIsEqual(llvm.TypeOf(arg), paramTypes[i]) {
			panic("type mismatch")
		}
		addr := llvm.BuildStructGEP2(c.builder, closureCtxType, x, uint(i), "closure_param_arg_addr")
		llvm.BuildStore(c.builder, arg, addr)
	}
	return x
}

func (c *Compiler) computeFunctionHash(fn *types.Func) uint32 {
	signature := fn.Type().(*types.Signature)
	hasher := fnv.New32a()
	io.WriteString(hasher, fn.Name())

	// Add the parameter types to the hash
	for i := 0; i < signature.Params().Len(); i++ {
		io.WriteString(hasher, signature.Params().At(i).Type().String())
	}

	// Add the return types to the hash
	for i := 0; i < signature.Results().Len(); i++ {
		io.WriteString(hasher, signature.Results().At(i).Type().String())
	}
	return hasher.Sum32()
}

func (c *Compiler) symbolName(pkg *types.Package, name string) string {
	path := "_"
	// pkg is nil for objects in Universe scope and possibly types
	// introduced via Eval (see also comment in object.sameId)
	if pkg != nil && pkg.Path() != "" {
		path = pkg.Path()
	}
	return path + "." + name
}
