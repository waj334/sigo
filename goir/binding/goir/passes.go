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

func NewAttachDebugInfoToConstantPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToConstantPass().ptr))
}

func NewAttachDebugInfoToFuncPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToFuncPass().ptr))
}

func NewAttachDebugInfoToGlobalPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoAttachDebugInfoToGlobalPass().ptr))
}

func NewCallPass() mlir.Pass {
	return mlir.WrapExternalPass(unsafe.Pointer(C.mlirCreateGoCallPass().ptr))
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
