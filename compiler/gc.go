package compiler

import (
	"context"
	"omibyte.io/sigo/llvm"
	"strings"
)

func (c *Compiler) GCPass(llvmCtx llvm.LLVMContextRef, module llvm.LLVMModuleRef) {
	ctx := context.WithValue(context.Background(), llvmContextKey{}, llvmCtx)

	var objects []llvm.LLVMValueRef
	for fn := llvm.GetFirstFunction(c.module); fn != nil; fn = llvm.GetNextFunction(fn) {
		for block := llvm.GetFirstBasicBlock(fn); block != nil; block = llvm.GetNextBasicBlock(block) {
			for inst := llvm.GetFirstInstruction(block); inst != nil; inst = llvm.GetNextInstruction(inst) {
				if llvm.IsACallInst(inst) != nil {
					calledFn := llvm.GetCalledValue(inst)
					calledFnName := llvm.GetValueName2(calledFn)
					if calledFnName == "runtime.gc_alloca" || calledFnName == "GCAlloc" {
						objects = append(objects, inst)
					}
				}
			}
		}
	}

	// Find the main.main function
	mainFn := llvm.GetNamedFunction(module, "main.main")

	// Insert gc_compact calls into every function reachable from main.main
	visited := make(map[llvm.LLVMValueRef]bool)
	c.insertComapctCalls(ctx, mainFn, visited)

}

func (c *Compiler) insertComapctCalls(ctx context.Context, fn llvm.LLVMValueRef, visited map[llvm.LLVMValueRef]bool) {
	fnName := llvm.GetValueName2(fn)
	if visited[fn] || strings.HasPrefix(fnName, "runtime.gc_") ||
		fnName == "Memcpy" ||
		fnName == "Memset" ||
		fnName == "Memmove" ||
		fnName == "Free" ||
		fnName == "Malloc" ||
		fnName == "GCAlloc" {
		return
	}
	visited[fn] = true

	// Visit called functions
	for block := llvm.GetFirstBasicBlock(fn); block != nil; block = llvm.GetNextBasicBlock(block) {
		for inst := llvm.GetFirstInstruction(block); inst != nil; inst = llvm.GetNextInstruction(inst) {
			if callInst := llvm.IsACallInst(inst); callInst != nil {
				calledFn := llvm.GetCalledValue(callInst)
				c.insertComapctCalls(ctx, calledFn, visited)
			}
		}
	}

	// Insert gc_compact call before the last instruction in the last basic block
	lastBlock := llvm.GetLastBasicBlock(fn)
	if lastBlock != nil {
		lastInstr := llvm.GetLastInstruction(lastBlock)
		if llvm.IsAReturnInst(lastInstr) != nil {
			llvm.PositionBuilderBefore(c.builder, lastInstr)
			c.createRuntimeCall(ctx, "gc_compact", []llvm.LLVMValueRef{})
		}
	}
}

func instructionComesBefore(a, b llvm.LLVMValueRef) bool {
	for inst := llvm.GetFirstInstruction(llvm.GetInstructionParent(a)); inst != nil; inst = llvm.GetNextInstruction(inst) {
		if inst == a {
			return true
		}
		if inst == b {
			return false
		}
	}
	return false
}
