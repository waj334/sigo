package runtime

import (
	"unsafe"

	"runtime/internal/allocator"
)

type stringDescriptor struct {
	array unsafe.Pointer
	len   int
}

func stringLen(descriptor unsafe.Pointer) int {
	str := (*stringDescriptor)(descriptor)
	return str.len
}

//go:export stringConcat runtime.stringConcat
func stringConcat(lhs unsafe.Pointer, rhs unsafe.Pointer) unsafe.Pointer {
	strLhs := (*stringDescriptor)(lhs)
	strRhs := (*stringDescriptor)(rhs)

	// Allocate storage buffer for the new string
	newLen := strLhs.len + strRhs.len
	newArray := allocator.GCAlloc(uintptr(newLen))

	// Copy the contents of the strings into the new buffer
	allocator.Memcpy(newArray, strLhs.array, uintptr(strLhs.len))
	allocator.Memcpy(unsafe.Add(newArray, strLhs.len), strRhs.array, uintptr(strRhs.len))

	// Return a new string
	result := stringDescriptor{
		array: newArray,
		len:   newLen,
	}

	return unsafe.Pointer(&result.array)
}

//go:export stringIndexAddr stringIndexAddr
func stringIndexAddr(s unsafe.Pointer, index int) unsafe.Pointer {
	str := (*stringDescriptor)(s)
	// Index MUST not be greater than the length of the string
	if index >= str.len {
		// TODO: Panic
		return nil
	}
	// Return the address of the element at the specified index
	return unsafe.Add(str.array, uintptr(index))
}

//go:export stringCompare runtime.stringCompare
func stringCompare(lhs, rhs unsafe.Pointer) bool {
	// TODO: This function can just call strncmp from the standard C library
	//  when it is able to be called.
	strLhs := *(*stringDescriptor)(lhs)
	strRhs := *(*stringDescriptor)(rhs)

	// Fast path
	if strLhs.len != strRhs.len {
		return false
	}

	// Compare each byte
	for i := 0; i < strLhs.len; i++ {
		charLhs := *(*byte)(unsafe.Pointer(uintptr(strLhs.array) + uintptr(i)))
		charRhs := *(*byte)(unsafe.Pointer(uintptr(strRhs.array) + uintptr(i)))
		if charLhs != charRhs {
			return false
		}
	}
	return true
}
