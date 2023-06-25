package compiler

import (
	"context"
	"go/types"
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
	t := llvm.FunctionType(returnType, argValueTypes, signature.Variadic())
	c.signatures[signature] = t

	// Return the type
	return t
}

func (c *Compiler) createClosure(ctx context.Context, signature *types.Signature, closureCtxType llvm.LLVMTypeRef, discardReturn bool) llvm.LLVMValueRef {
	c.createType(ctx, signature)
	fnType := c.signatures[signature]

	// Get the return type of the function to be called by this closure
	returnType := llvm.GetReturnType(fnType)
	if discardReturn {
		returnType = llvm.VoidTypeInContext(c.currentContext(ctx))
	}
	closureFnType := llvm.FunctionType(
		returnType,
		[]llvm.LLVMTypeRef{c.ptrType.valueType},
		false)
	linkname := c.symbolName(c.currentPackage(ctx).Pkg, "closure")
	closureFnValue := llvm.AddFunction(c.module, linkname, closureFnType)
	llvm.SetSection(closureFnValue, ".text."+llvm.GetValueName2(closureFnValue))

	currentBlock := llvm.GetInsertBlock(c.builder)
	defer llvm.PositionBuilderAtEnd(c.builder, currentBlock)

	block := llvm.AppendBasicBlockInContext(c.currentContext(ctx), closureFnValue, "closure_entry")
	llvm.PositionBuilderAtEnd(c.builder, block)
	params := llvm.GetParam(closureFnValue, 0)
	params = llvm.BuildLoad2(c.builder, closureCtxType, params, "")

	// Load the function pointer
	fnPtr := llvm.BuildExtractValue(c.builder, params, 0, "")

	// Load each of the function parameter values
	var args []llvm.LLVMValueRef
	for i := uint(1); i < llvm.CountStructElementTypes(llvm.TypeOf(params)); i++ {
		arg := llvm.BuildExtractValue(c.builder, params, i, "")
		args = append(args, arg)
	}

	// Build the call
	ret := llvm.BuildCall2(c.builder, fnType, fnPtr, args, "")
	if returnType == llvm.VoidTypeInContext(c.currentContext(ctx)) || discardReturn {
		llvm.BuildRetVoid(c.builder)
	} else {
		llvm.BuildRet(c.builder, ret)
	}

	return closureFnValue
}

func (c *Compiler) createClosureContextType(ctx context.Context, signature *types.Signature) llvm.LLVMTypeRef {
	c.createType(ctx, signature)
	fnType := c.signatures[signature]
	paramTypes := llvm.GetParamTypes(fnType)
	paramTypes = append([]llvm.LLVMTypeRef{c.ptrType.valueType}, paramTypes...)
	paramsStructType := llvm.StructTypeInContext(c.currentContext(ctx), paramTypes, false)
	return paramsStructType
}

func (c *Compiler) createClosureContext(ctx context.Context, closureCtxType llvm.LLVMTypeRef, fn llvm.LLVMValueRef, args []llvm.LLVMValueRef) llvm.LLVMValueRef {
	paramTypes := llvm.GetStructElementTypes(closureCtxType)

	// Prepend the function pointer to the args
	args = append([]llvm.LLVMValueRef{fn}, args...)

	// Fail early if the number arguments doesn't match the number function parameters
	if len(args) != len(paramTypes) {
		panic("len(args) != len(params)")
	}

	// Create an instance of the params struct
	agg := llvm.GetUndef(closureCtxType)
	for i, arg := range args {
		// Verify argument types against param types
		if !llvm.TypeIsEqual(llvm.TypeOf(arg), paramTypes[i]) {
			panic("type mismatch")
		}
		agg = llvm.BuildInsertValue(c.builder, agg, arg, uint(i), "")
	}

	// Return the address to the struct
	return c.addressOf(ctx, agg)
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
