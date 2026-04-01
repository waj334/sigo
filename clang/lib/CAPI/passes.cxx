#include "go-clang/capi/passes.h"

#include <clang/CIR/Dialect/Passes.h>
#include <clang/CIR/LowerToLLVM.h>
#include <clang/CIR/Passes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>

//===----------------------------------------------------------------------===//
// Individual pass factories
//===----------------------------------------------------------------------===//

MlirPass goClangCreateCIRCanonicalizePass() {
  return wrap(mlir::createCIRCanonicalizePass().release());
}

MlirPass goClangCreateCIRFlattenCFGPass() {
  return wrap(mlir::createCIRFlattenCFGPass().release());
}

MlirPass goClangCreateCIRSimplifyPass() {
  return wrap(mlir::createCIRSimplifyPass().release());
}

MlirPass goClangCreateCXXABILoweringPass() {
  return wrap(mlir::createCXXABILoweringPass().release());
}

MlirPass goClangCreateHoistAllocasPass() {
  return wrap(mlir::createHoistAllocasPass().release());
}

MlirPass goClangCreateLoweringPreparePass() {
  return wrap(mlir::createLoweringPreparePass().release());
}

MlirPass goClangCreateGotoSolverPass() {
  return wrap(mlir::createGotoSolverPass().release());
}

MlirPass goClangCreateConvertCIRToLLVMPass() {
  return wrap(cir::direct::createConvertCIRToLLVMPass().release());
}

//===----------------------------------------------------------------------===//
// Pipeline helpers
//===----------------------------------------------------------------------===//

void goClangPopulateCIRPreLoweringPasses(MlirOpPassManager pm) {
  mlir::populateCIRPreLoweringPasses(*unwrap(pm));
}

void goClangPopulateCIRToLLVMPasses(MlirOpPassManager pm) {
  cir::direct::populateCIRToLLVMPasses(*unwrap(pm));
}

//===----------------------------------------------------------------------===//
// Direct CIR → LLVM IR lowering
//===----------------------------------------------------------------------===//

LLVMModuleRef goClangLowerCIRToLLVMIR(MlirModule module,
                                       LLVMContextRef llvmCtx) {
  auto moduleOp = unwrap(module);
  auto *ctx = llvm::unwrap(llvmCtx);
  // Register the CIR → LLVM IR translation interface so the MLIR translator
  // knows how to handle CIR-specific attributes and operations.
  mlir::registerCIRDialectTranslation(*moduleOp->getContext());
  auto llvmModule = cir::direct::lowerDirectlyFromCIRToLLVMIR(moduleOp, *ctx);
  if (!llvmModule)
    return nullptr;
  return llvm::wrap(llvmModule.release());
}
