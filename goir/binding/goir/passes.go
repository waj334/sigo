package goir

/*
#include <Go-c/mlir/Passes.h>
*/
import "C"
import (
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

func RegisterGoPasses() {
	C.mlirRegisterGoPasses()
}

func NewAttachDebugInfoToAllocaPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToAllocaPass().ptr))
}

func NewAttachDebugInfoToFuncPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToFuncPass().ptr))
}

func NewAttachDebugInfoToGlobalPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToGlobalPass().ptr))
}

func NewAttachDebugInfoToLLVMFuncPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToLLVMFuncPass().ptr))
}

func NewCallPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoCallPass().ptr))
}

func NewEliminateRedundantNilChecksPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoEliminateRedundantNilChecksPass().ptr))
}

func NewFuncPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoFuncPass().ptr))
}

func NewGlobalConstantsPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoGlobalConstantsPass().ptr))
}

func NewGlobalInitializerPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoGlobalInitializerPass().ptr))
}

func NewHeapEscapePass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoHeapEscapePass().ptr))
}

func NewInsertGCWriteBarrierPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoInsertGCWriteBarrierPass().ptr))
}

func NewLowerToCorePass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoLowerToCorePass().ptr))
}

func NewLowerToLLVMPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoLowerToLLVMPass().ptr))
}

func NewPreprocessingPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoPreprocessingPass().ptr))
}

func NewValueNormalizationFuncPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoValueNormalizationFuncPass().ptr))
}

func NewValueNormalizationGlobalPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoValueNormalizationGlobalPass().ptr))
}
