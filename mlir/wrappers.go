package mlir

/*
#include <stdlib.h>
#include "mlir-c/IR.h"
#include <string.h>
*/
import "C"
import "unsafe"

func createStringRef(input string) C.MlirStringRef {
	sz := C.size_t(len(input))
	cstr := C.malloc(sz)
	gstr := unsafe.Pointer(unsafe.StringData(input))

	// Copy the bytes from the Go string into the C String.
	C.memcpy(cstr, gstr, sz)

	// Create and return the string reference.
	return C.mlirStringRefCreate((*C.char)(cstr), sz)
}

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
