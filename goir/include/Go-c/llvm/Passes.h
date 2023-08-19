#ifndef GO_PASSES_C_H
#define GO_PASSES_C_H

#include <llvm-c/Transforms//PassBuilder.h>
#include <llvm-c/Core.h>
#include <llvm-c/Error.h>
#include <llvm-c/TargetMachine.h>

#ifdef __cplusplus
extern "C" {
#endif

void runGoLanguagePasses(LLVMModuleRef M, LLVMTargetMachineRef TM, uint64_t allocSizeThreshold);

#ifdef __cplusplus
}
#endif

#endif // GO_PASSES_C_H