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

	target.SetSwigcptr(uintptr(unsafe.Pointer(CTarget)))
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
		*(*C.LLVMValueRef)(unsafe.Pointer(uintptr(unsafe.Pointer(edgesArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMValueRef)(unsafe.Pointer(edge.cptr))
		*(*C.LLVMBasicBlockRef)(unsafe.Pointer(uintptr(unsafe.Pointer(blocksArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMBasicBlockRef)(unsafe.Pointer(blocks[i].cptr))
	}

	C.LLVMAddIncoming(C.LLVMValueRef(unsafe.Pointer(phi.cptr)), edgesArr, blocksArr, C.uint(len(edges)))
}

func GetParamTypes(FunctionTy LLVMTypeRef) (result []LLVMTypeRef) {
	count := int(CountParamTypes(FunctionTy))
	output := make([]C.LLVMTypeRef, count)
	if count > 0 {
		C.LLVMGetParamTypes(C.LLVMTypeRef(unsafe.Pointer(FunctionTy.cptr)), (*C.LLVMTypeRef)(unsafe.Pointer(&output[0])))
	}
	result = make([]LLVMTypeRef, 0, count)
	for _, value := range output {
		var ty LLVMTypeRef
		ty.SetSwigcptr(uintptr(unsafe.Pointer(value)))
		result = append(result, ty)
	}
	return
}

func GetParams(FunctionValue LLVMValueRef) (result []LLVMValueRef) {
	count := int(CountParams(FunctionValue))
	output := make([]C.LLVMValueRef, count)
	if count > 0 {
		C.LLVMGetParams(C.LLVMValueRef(unsafe.Pointer(FunctionValue.cptr)), (*C.LLVMValueRef)(unsafe.Pointer(&output[0])))
	}
	result = make([]LLVMValueRef, 0, count)
	for _, value := range output {
		var val LLVMValueRef
		val.SetSwigcptr(uintptr(unsafe.Pointer(value)))
		result = append(result, val)
	}
	return
}

func GetStructElementTypes(StructTy LLVMTypeRef) (result []LLVMTypeRef) {
	count := int(CountStructElementTypes(StructTy))
	output := make([]C.LLVMTypeRef, count)
	if count > 0 {
		C.LLVMGetStructElementTypes(C.LLVMTypeRef(unsafe.Pointer(StructTy.cptr)), (*C.LLVMTypeRef)(unsafe.Pointer(&output[0])))
	}
	result = make([]LLVMTypeRef, 0, count)
	for _, value := range output {
		var ty LLVMTypeRef
		ty.SetSwigcptr(uintptr(unsafe.Pointer(value)))
		result = append(result, ty)
	}
	return
}

func TypeIsEqual(a, b LLVMTypeRef) bool {
	return a.cptr == b.cptr
}

func TargetMachineEmitToFile2(machine LLVMTargetMachineRef, module LLVMModuleRef, output string, fileType LLVMCodeGenFileType) (ok bool, errMsg string) {
	COutput := C.CString(output)
	defer C.free(unsafe.Pointer(COutput))

	var CErrStr *C.char

	result := C.LLVMTargetMachineEmitToFile(
		C.LLVMTargetMachineRef(unsafe.Pointer(machine.cptr)),
		C.LLVMModuleRef(unsafe.Pointer(module.cptr)),
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
	result := C.LLVMVerifyModule(C.LLVMModuleRef(unsafe.Pointer(module.cptr)), C.LLVMVerifierFailureAction(failureAction), &CErrStr)
	ok = result == 0
	if CErrStr != nil {
		errMsg = C.GoString(CErrStr)
		C.LLVMDisposeMessage(CErrStr)
	}
	return
}

func GetValueName2(value LLVMValueRef) string {
	var length C.size_t
	nameStr := C.LLVMGetValueName2(C.LLVMValueRef(unsafe.Pointer(value.cptr)), (*C.size_t)(&length))
	return C.GoStringN(nameStr, C.int(length))
}
