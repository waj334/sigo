package clang

/*
#include <go-clang/capi/passes.h>
#include <llvm-c/Core.h>
*/
import "C"
import (
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

//===----------------------------------------------------------------------===//
// Individual CIR pass factories
//===----------------------------------------------------------------------===//

// NewCIRCanonicalizePass returns a pass that eliminates redundant branches
// and removes empty scopes.
func NewCIRCanonicalizePass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateCIRCanonicalizePass().ptr)
}

// NewCIRFlattenCFGPass returns a pass that flattens the CIR control-flow
// graph.
func NewCIRFlattenCFGPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateCIRFlattenCFGPass().ptr)
}

// NewCIRSimplifyPass returns a pass that runs algebraic simplification
// patterns over CIR operations.
func NewCIRSimplifyPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateCIRSimplifyPass().ptr)
}

// NewCXXABILoweringPass returns a pass that lowers C++ ABI-specific CIR
// operations.
func NewCXXABILoweringPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateCXXABILoweringPass().ptr)
}

// NewHoistAllocasPass returns a pass that hoists alloca operations to the
// function entry block.
func NewHoistAllocasPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateHoistAllocasPass().ptr)
}

// NewLoweringPreparePass returns a pass that prepares CIR for lowering
// (no-ASTContext variant).
func NewLoweringPreparePass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateLoweringPreparePass().ptr)
}

// NewGotoSolverPass returns a pass that resolves goto statements.
func NewGotoSolverPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateGotoSolverPass().ptr)
}

// NewConvertCIRToLLVMPass returns a pass that converts all CIR ops to the
// LLVM dialect.
func NewConvertCIRToLLVMPass() mlir.Pass {
	return mlir.WrapExternalPass(C.goClangCreateConvertCIRToLLVMPass().ptr)
}

//===----------------------------------------------------------------------===//
// Pipeline helpers
//===----------------------------------------------------------------------===//

// PopulateCIRPreLoweringPasses adds the CIR pre-lowering pipeline to pm:
// HoistAllocas → FlattenCFG → GotoSolver.
func PopulateCIRPreLoweringPasses(pm mlir.OpPassManager) {
	C.goClangPopulateCIRPreLoweringPasses(unwrapOpPassManager(pm))
}

// PopulateCIRToLLVMPasses adds the full CIR → LLVM dialect conversion
// pipeline to pm.
func PopulateCIRToLLVMPasses(pm mlir.OpPassManager) {
	C.goClangPopulateCIRToLLVMPasses(unwrapOpPassManager(pm))
}

//===----------------------------------------------------------------------===//
// Direct CIR → LLVM IR lowering
//===----------------------------------------------------------------------===//

// LowerCIRToLLVMIR lowers a CIR module directly to an LLVM IR module.
// The caller owns the returned LLVMModuleRef.
// Returns a null ref on failure.
func LowerCIRToLLVMIR(module mlir.Module, llvmCtx mlir.LLVMContextRef) mlir.LLVMModuleRef {
	raw := C.goClangLowerCIRToLLVMIR(
		unwrapModule(module),
		C.LLVMContextRef(llvmCtx.Ptr()),
	)
	if raw == nil {
		return mlir.LLVMModuleRef{}
	}
	return mlir.WrapExternalLLVMModuleRef(unsafe.Pointer(raw))
}
