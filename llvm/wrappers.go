// This file contains wrappers that are too difficult for SWIG to handle auto-magically

package llvm

/*
#include <stdlib.h>
#include "llvm-c/Analysis.h"
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
*/
import "C"
import "unsafe"

type (
	LLVMTypeKind      int
	LLVMIntPredicate  int
	LLVMRealPredicate int
)

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

const (
	LLVMIntEQ  LLVMIntPredicate = iota + 32 // equal
	LLVMIntNE                               // not equal
	LLVMIntUGT                              // unsigned greater than
	LLVMIntUGE                              // unsigned greater or equal
	LLVMIntULT                              // unsigned less than
	LLVMIntULE                              // unsigned less or equal
	LLVMIntSGT                              // signed greater than
	LLVMIntSGE                              // signed greater or equal
	LLVMIntSLT                              // signed less than
	LLVMIntSLE                              // signed less or equal
)

const (
	LLVMRealPredicateFalse = iota // Always false (always folded)
	LLVMRealOEQ                   // True if ordered and equal
	LLVMRealOGT                   // True if ordered and greater than
	LLVMRealOGE                   // True if ordered and greater than or equal
	LLVMRealOLT                   // True if ordered and less than
	LLVMRealOLE                   // True if ordered and less than or equal
	LLVMRealONE                   // True if ordered and operands are unequal
	LLVMRealORD                   // True if ordered (no nans)
	LLVMRealUNO                   // True if unordered: isnan(X) | isnan(Y)
	LLVMRealUEQ                   // True if unordered or equal
	LLVMRealUGT                   // True if unordered or greater than
	LLVMRealUGE                   // True if unordered, greater than, or equal
	LLVMRealULT                   // True if unordered or less than
	LLVMRealULE                   // True if unordered, less than, or equal
	LLVMRealUNE                   // True if unordered or not equal
	LLVMRealPredicateTrue         // Always true (always folded)
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

// AddIncoming adds an incoming value to the end of a PHI list.
func AddIncoming(phi LLVMValueRef, edges []LLVMValueRef, blocks []LLVMBasicBlockRef) {
	if len(edges) != len(blocks) {
		panic("there must be as many edges as blocks")
	}

	edgesArr := (*C.LLVMValueRef)(C.malloc(C.size_t(len(edges)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	blocksArr := (*C.LLVMBasicBlockRef)(C.malloc(C.size_t(len(edges)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(edgesArr))
	defer C.free(unsafe.Pointer(blocksArr))

	for i, edge := range edges {
		*(*C.LLVMValueRef)(unsafe.Pointer(uintptr(unsafe.Pointer(edgesArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMValueRef)(unsafe.Pointer(edge.Swigcptr()))
		*(*C.LLVMBasicBlockRef)(unsafe.Pointer(uintptr(unsafe.Pointer(blocksArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMBasicBlockRef)(unsafe.Pointer(blocks[i].Swigcptr()))
	}

	C.LLVMAddIncoming(C.LLVMValueRef(unsafe.Pointer(phi.Swigcptr())), edgesArr, blocksArr, C.uint(len(edges)))
}

func GetParamTypes(FunctionTy LLVMTypeRef) (result []LLVMTypeRef) {
	count := int(CountParamTypes(FunctionTy))
	output := make([]C.LLVMTypeRef, count)
	if count > 0 {
		C.LLVMGetParamTypes(C.LLVMTypeRef(unsafe.Pointer(FunctionTy.Swigcptr())), (*C.LLVMTypeRef)(unsafe.Pointer(&output[0])))
	}
	result = make([]LLVMTypeRef, 0, count)
	for _, value := range output {
		result = append(result, SwigcptrLLVMTypeRef(unsafe.Pointer(value)))
	}
	return
}

func GetStructElementTypes(StructTy LLVMTypeRef) (result []LLVMTypeRef) {
	count := int(CountStructElementTypes(StructTy))
	output := make([]C.LLVMTypeRef, count)
	if count > 0 {
		C.LLVMGetStructElementTypes(C.LLVMTypeRef(unsafe.Pointer(StructTy.Swigcptr())), (*C.LLVMTypeRef)(unsafe.Pointer(&output[0])))
	}
	result = make([]LLVMTypeRef, 0, count)
	for _, value := range output {
		result = append(result, SwigcptrLLVMTypeRef(unsafe.Pointer(value)))
	}
	return
}

func TypeIsEqual(a, b LLVMTypeRef) bool {
	return a.Swigcptr() == b.Swigcptr()
}

func TargetMachineEmitToFile2(machine LLVMTargetMachineRef, module LLVMModuleRef, output string, fileType LLVMCodeGenFileType) (ok bool, errMsg string) {
	COutput := C.CString(output)
	defer C.free(unsafe.Pointer(COutput))

	var CErrStr *C.char

	result := C.LLVMTargetMachineEmitToFile(
		C.LLVMTargetMachineRef(unsafe.Pointer(machine.Swigcptr())),
		C.LLVMModuleRef(unsafe.Pointer(module.Swigcptr())),
		COutput,
		C.LLVMCodeGenFileType(fileType),
		&CErrStr)

	ok = result == 0
	if CErrStr != nil {
		errMsg = C.GoString(CErrStr)
		C.LLVMDisposeMessage(CErrStr)
	}
	return
}

func VerifyModule2(module LLVMModuleRef, failureAction LLVMVerifierFailureAction) (ok bool, errMsg string) {
	var CErrStr *C.char
	result := C.LLVMVerifyModule(C.LLVMModuleRef(unsafe.Pointer(module.Swigcptr())), C.LLVMVerifierFailureAction(failureAction), &CErrStr)
	ok = result == 0
	if CErrStr != nil {
		errMsg = C.GoString(CErrStr)
		C.LLVMDisposeMessage(CErrStr)
	}
	return
}

func GetValueName2(value LLVMValueRef) string {
	var length C.size_t
	nameStr := C.LLVMGetValueName2(C.LLVMValueRef(unsafe.Pointer(value.Swigcptr())), (*C.size_t)(&length))
	return C.GoStringN(nameStr, C.int(length))
}
