#ifndef GO_CLANG_CAPI_PASSES_H
#define GO_CLANG_CAPI_PASSES_H

#include <llvm-c/Core.h>
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Individual CIR dialect pass factories.
// Each returns an owned MlirPass. Add to a PassManager with
// mlirPassManagerAddOwnedPass / mlirOpPassManagerAddOwnedPass.
//===----------------------------------------------------------------------===//

// Eliminates redundant branches and removes empty scopes.
MlirPass goClangCreateCIRCanonicalizePass();

// Flattens the CIR control-flow graph.
MlirPass goClangCreateCIRFlattenCFGPass();

// Runs algebraic simplification patterns over CIR operations.
MlirPass goClangCreateCIRSimplifyPass();

// Lowers C++ ABI-specific CIR operations.
MlirPass goClangCreateCXXABILoweringPass();

// Hoists alloca operations to the function entry block.
MlirPass goClangCreateHoistAllocasPass();

// Prepares CIR for lowering (no ASTContext variant).
MlirPass goClangCreateLoweringPreparePass();

// Resolves goto statements.
MlirPass goClangCreateGotoSolverPass();

// Creates the pass that converts all CIR ops to the LLVM dialect.
MlirPass goClangCreateConvertCIRToLLVMPass();

//===----------------------------------------------------------------------===//
// Pipeline helpers — add a group of passes to an existing OpPassManager.
//===----------------------------------------------------------------------===//

// Adds the CIR pre-lowering pipeline: HoistAllocas → FlattenCFG → GotoSolver.
void goClangPopulateCIRPreLoweringPasses(MlirOpPassManager pm);

// Adds the full CIR → LLVM dialect conversion pipeline.
void goClangPopulateCIRToLLVMPasses(MlirOpPassManager pm);

//===----------------------------------------------------------------------===//
// Direct CIR → LLVM IR lowering.
//===----------------------------------------------------------------------===//

// Lowers a CIR module directly to an LLVM IR module.
// The returned LLVMModuleRef is owned by the caller; free with LLVMDisposeModule.
// Returns null on failure.
LLVMModuleRef goClangLowerCIRToLLVMIR(MlirModule module, LLVMContextRef llvmCtx);

#ifdef __cplusplus
}
#endif

#endif // GO_CLANG_CAPI_PASSES_H
