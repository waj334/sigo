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

// NewUnrealizedConversionCastOp creates a builtin.unrealized_conversion_cast
// that casts value to the given target type. The resulting op can be used as a
// dialect-boundary placeholder; it is eliminated once both sides lower to the
// same concrete type.
func NewUnrealizedConversionCastOp(ctx mlir.Context, targetType mlir.TypeLike, value mlir.ValueLike, loc mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirCreateUnrealizedConversionCastOp(
		unwrapContext(ctx),
		unwrapType(targetType),
		unwrapValue(value),
		unwrapLocation(loc),
	))
}
