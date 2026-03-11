package goir

/*
#include <Go-c/mlir/Dialects.h>
#include <Go-c/mlir/Enums.h>
*/
import "C"
import (
	"unsafe"

	"golang.org/x/exp/constraints"
	"pkg.si-go.dev/go-mlir/mlir"
)

func unwrapAttribute(attr mlir.AttributeLike) C.MlirAttribute {
	if attr == nil {
		return C.MlirAttribute{}
	}
	return C.MlirAttribute{ptr: attr.Ptr()}
}

func unwrapAttributeSlice[T mlir.AttributeLike](attrs []T) *C.MlirAttribute {
	if len(attrs) == 0 {
		return nil
	}

	rawAttrs := make([]C.MlirAttribute, len(attrs))
	for i, a := range attrs {
		rawAttrs[i] = unwrapAttribute(a)
	}
	return &rawAttrs[0]
}

func unwrapBlock(block mlir.Block) C.MlirBlock {
	return C.MlirBlock{ptr: block.Ptr()}
}

func unwrapContext(ctx mlir.Context) C.MlirContext {
	return C.MlirContext{ptr: ctx.Ptr()}
}

func unwrapLocation(loc mlir.LocationLike) C.MlirLocation {
	if loc == nil {
		return C.MlirLocation{}
	}
	return C.MlirLocation{ptr: loc.Ptr()}
}

func unwrapLLVMTargetDataRef(ref mlir.LLVMTargetDataRef) C.LLVMTargetDataRef {
	return C.LLVMTargetDataRef(ref.Ptr())
}

func unwrapModule(module mlir.Module) C.MlirModule {
	return C.MlirModule{ptr: module.Ptr()}
}

func unwrapOperation(op mlir.Operation) C.MlirOperation {
	return C.MlirOperation{ptr: op.Ptr()}
}

type primitives interface {
	constraints.Integer | constraints.Float
}

func unwrapRegion(region mlir.Region) C.MlirRegion {
	return C.MlirRegion{ptr: region.Ptr()}
}

func unwrapStringRef(ref mlir.StringRef) C.MlirStringRef {
	return C.MlirStringRef{
		data:   (*C.char)(ref.Data()),
		length: C.size_t(ref.Length()),
	}
}

func unwrapStringSlice(values []string) *C.MlirStringRef {
	if len(values) == 0 {
		return nil
	}

	c := make([]C.MlirStringRef, len(values))
	for i, value := range values {
		c[i] = unwrapStringRef(mlir.NewStringRef(value))
	}
	return &c[0]
}

func unwrapType(typ mlir.TypeLike) C.MlirType {
	if typ == nil {
		return C.MlirType{}
	}
	return C.MlirType{ptr: typ.Ptr()}
}

func unwrapTypeSlice[T mlir.TypeLike](types []T) *C.MlirType {
	if len(types) == 0 {
		return nil
	}

	rawTypes := make([]C.MlirType, len(types))
	for i, t := range types {
		rawTypes[i] = unwrapType(t)
	}
	return &rawTypes[0]
}

func unwrapValue(val mlir.ValueLike) C.MlirValue {
	if val == nil {
		return C.MlirValue{}
	}
	return C.MlirValue{ptr: val.Ptr()}
}

func unwrapValueSlice[T mlir.ValueLike](values []T) *C.MlirValue {
	if len(values) == 0 {
		return nil
	}

	rawAttrs := make([]C.MlirValue, len(values))
	for i, v := range values {
		rawAttrs[i] = unwrapValue(v)
	}
	return &rawAttrs[0]
}

func wrapAttribute(raw C.MlirAttribute) mlir.Attribute {
	return mlir.WrapExternalAttribute(unsafe.Pointer(raw.ptr))
}

func wrapOperation(raw C.MlirOperation) mlir.Operation {
	return mlir.WrapExternalOperation(unsafe.Pointer(raw.ptr))
}

func wrapType(raw C.MlirType) mlir.Type {
	return mlir.WrapExternalType(unsafe.Pointer(raw.ptr))
}
