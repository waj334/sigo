package goir

/*
#include <Go-c/mlir/Dialects.h>
*/
import "C"
import (
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

func DialectHandle() mlir.DialectHandle {
	return mlir.WrapExternalDialectHandle(unsafe.Pointer(C.mlirGoDialectHandleGet().ptr))
}

func BindRuntimeType(module mlir.Module, mnemonic string, runtimeType mlir.TypeLike) {
	refMnemonic := mlir.NewStringRef(mnemonic)
	defer refMnemonic.Destroy()
	C.mlirGoBindRuntimeType(
		unwrapModule(module),
		unwrapStringRef(refMnemonic),
		unwrapType(runtimeType),
	)
}

func BindRuntimeTypeToType(module mlir.Module, primitiveType mlir.TypeLike, runtimeType mlir.TypeLike) {
	C.mlirGoBindRuntimeTypeToType(
		unwrapModule(module),
		unwrapType(primitiveType),
		unwrapType(runtimeType),
	)
}

func DumpTail(block mlir.Block, count int) {
	C.mlirGoBlockDumpTail(
		unwrapBlock(block),
		C.int(count),
	)
}

func TypeHash(typ mlir.TypeLike) int {
	return int(C.mlirTypeHash(unwrapType(typ)))
}

func SetTargetDataLayout(module mlir.Module, layout mlir.LLVMTargetDataRef) {
	C.mlirGoSetTargetDataLayout(
		unwrapModule(module),
		unwrapLLVMTargetDataRef(layout),
	)
}

func SetTargetTriple(module mlir.Module, triple string) {
	refTriple := mlir.NewStringRef(triple)
	defer refTriple.Destroy()
	C.mlirGoSetTargetTriple(
		unwrapModule(module),
		unwrapStringRef(refTriple),
	)
}
