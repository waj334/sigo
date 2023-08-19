package mlir

/*
#include <stdlib.h>
#include "mlir-c/IR.h"
*/
import "C"

/*
func BlockCreate2(args []Type, locs []Location) Block {
	if len(args) != len(locs) {
		panic("len(args) != len(locs)")
	}

	argsArr := (*C.MlirType)(C.malloc(C.size_t(len(args)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	locsArr := (*C.MlirLocation)(C.malloc(C.size_t(len(locs)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(argsArr))
	defer C.free(unsafe.Pointer(locsArr))

	for i, arg := range args {
		*(*C.MlirType)(unsafe.Pointer(uintptr(unsafe.Pointer(argsArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = *(*C.MlirType)(unsafe.Pointer(arg.Swigcptr()))
		*(*C.MlirLocation)(unsafe.Pointer(uintptr(unsafe.Pointer(locsArr)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = *(*C.MlirLocation)(unsafe.Pointer(locs[i].Swigcptr()))
	}

	val := C.mlirBlockCreate(C.intptr_t(len(args)), argsArr, locsArr)
	result := (Block)(SwigcptrBlock(*(*uintptr)(unsafe.Pointer(&val))))
	if Swig_escape_always_false {
		Swig_escape_val = result
	}
	return result
}
*/

func GoCreateGepOperation2(ctx Context, base Value, baseType Type, indices []any, resultType Type, location Location) Operation {
	var dynamicIndices []Value
	var constIndices []int32
	i := 0
	for _, index := range indices {
		switch index := index.(type) {
		case int:
			constIndices = append(constIndices, int32(index))
		case int8:
			constIndices = append(constIndices, int32(index))
		case int16:
			constIndices = append(constIndices, int32(index))
		case int32:
			constIndices = append(constIndices, index) // already int32
		case int64:
			constIndices = append(constIndices, int32(index))
		case uint:
			constIndices = append(constIndices, int32(index))
		case uint8:
			constIndices = append(constIndices, int32(index))
		case uint16:
			constIndices = append(constIndices, int32(index))
		case uint32:
			constIndices = append(constIndices, int32(index))
		case uint64:
			constIndices = append(constIndices, int32(index))
		case Value:
			constIndices = append(constIndices, int32(i|0x8000_0000))
			dynamicIndices = append(dynamicIndices, index)
			i++
		default:
			panic("unexpected index type")
		}
	}

	return GoCreateGepOperation(ctx, base, baseType, constIndices, dynamicIndices, resultType, location)
}
