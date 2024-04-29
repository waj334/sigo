
#include "Go-c/llvm/Passes.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/CBindingWrapping.h>

using namespace llvm;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine,
                                   LLVMTargetMachineRef)

void runGoLanguagePasses(LLVMModuleRef M, LLVMTargetMachineRef TM, uint64_t allocSizeThreshold)
{
    TargetMachine *Machine = unwrap(TM);

    Module *Mod = unwrap(M);
    PassInstrumentationCallbacks PIC;
    PassBuilder PB(Machine, PipelineTuningOptions(), std::nullopt, &PIC);
    ModulePassManager MPM;
    FunctionPassManager FPM;

    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerLoopAnalyses(LAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerModuleAnalyses(MAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    StandardInstrumentations SI(Mod->getContext(), false, true);
    SI.registerCallbacks(PIC, &MAM);
    MPM.addPass(VerifierPass());

    // Add Go language specific passes

    // Add the function pass manager to the module pass manager
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

    MPM.run(*Mod, MAM);
}