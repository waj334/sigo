// This file contains wrappers that are too difficult for SWIG to handle auto-magically

package llvm

/*
#include <stdlib.h>
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
*/
import "C"
import "unsafe"

type LLVMTypeKind int

const (
	VoidTypeKind LLVMTypeKind = iota
	HalfTypeKind
	FloatTypeKind
	DoubleTypeKind
	X86_FP80TypeKind
	FP128TypeKind
	PPC_FP128TypeKind
	LabelTypeKind
	IntegerTypeKind
	FunctionTypeKind
	StructTypeKind
	ArrayTypeKind
	PointerTypeKind
	VectorTypeKind
	MetadataTypeKind
	X86_MMXTypeKind
	TokenTypeKind
	ScalableVectorTypeKind
	BFloatTypeKind
	X86_AMXTypeKind
	TargetExtTypeKind
)

// GetTargetFromTriple Finds the target corresponding to the given triple and stores it in T.
func GetTargetFromTriple(triple string) (target LLVMTargetRef, errStr string, ok bool) {
	CTriple := C.CString(triple)
	defer C.free(unsafe.Pointer(CTriple))

	var CTarget C.LLVMTargetRef
	var CErrStr *C.char

	result := C.LLVMGetTargetFromTriple(CTriple, &CTarget, &CErrStr)
	ok = result == 0
	if CErrStr != nil {
		errStr = C.GoString(CErrStr)
		C.LLVMDisposeMessage(CErrStr)
	}

	target = SwigcptrLLVMTargetRef(unsafe.Pointer(CTarget))
	return
}
