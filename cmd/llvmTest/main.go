package main

import "omibyte.io/sigo/llvm"

func main() {
	context := llvm.ContextCreate()
	module := llvm.ModuleCreateWithNameInContext("hello", context)
	builder := llvm.CreateBuilderInContext(context)

	int8Type := llvm.Int8TypeInContext(context)
	int8PtrType := llvm.PointerType(int8Type, 0)
	int32Type := llvm.Int32TypeInContext(context)

	putsArgs := []llvm.LLVMTypeRef{
		int8Type,
	}

	putsFnType := llvm.FunctionType(int32Type, putsArgs, false)
	putsFn := llvm.AddFunction(module, "puts", putsFnType)

	mainFnType := llvm.FunctionType(int32Type, nil, false)
	mainFn := llvm.AddFunction(module, "main", mainFnType)

	entry := llvm.AppendBasicBlockInContext(context, mainFn, "entry")
	llvm.PositionBuilderAtEnd(builder, entry)

	putsFnArgs := []llvm.LLVMValueRef{
		llvm.BuildPointerCast(builder,
			llvm.BuildGlobalString(builder, "Hello, World...bih...", "G"),
			int8PtrType, "0"),
	}

	llvm.BuildCall2(builder, putsFnType, putsFn, putsFnArgs, "i")
	llvm.BuildRet(builder, llvm.ConstInt(int32Type, 0, false))

	llvm.PrintModuleToFile(module, "hello.ll", nil)

	llvm.DisposeBuilder(builder)
	llvm.DisposeModule(module)
	llvm.ContextDispose(context)
}
