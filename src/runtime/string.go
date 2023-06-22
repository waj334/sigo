package runtime

import (
	"unsafe"
)

type stringDescriptor struct {
	array unsafe.Pointer
	len   int
}

func stringLen(descriptor *stringDescriptor) int {
	str := descriptor
	return str.len
}

//go:export stringConcat runtime.stringConcat
func stringConcat(lhs *stringDescriptor, rhs *stringDescriptor) stringDescriptor {
	// Allocate storage buffer for the new string
	newLen := lhs.len + rhs.len
	newArray := alloc(uintptr(newLen))

	// Copy the contents of the strings into the new buffer
	memcpy(newArray, lhs.array, uintptr(lhs.len))
	memcpy(unsafe.Add(newArray, lhs.len), rhs.array, uintptr(rhs.len))

	// Return a new string
	return stringDescriptor{
		array: newArray,
		len:   newLen,
	}
}

//go:export stringIndexAddr stringIndexAddr
func stringIndexAddr(str *stringDescriptor, index int) unsafe.Pointer {
	// Index MUST not be greater than the length of the string
	if index >= str.len {
		panic("runtime: index out of range")
	}
	// Return the address of the element at the specified index
	return unsafe.Add(str.array, uintptr(index))
}

//go:export stringCompare runtime.stringCompare
func stringCompare(lhs, rhs *stringDescriptor) bool {
	// Fast path
	if lhs.len != rhs.len {
		return false
	}

	// Compare each byte
	for i := 0; i < lhs.len; i++ {
		charLhs := *(*byte)(unsafe.Pointer(uintptr(lhs.array) + uintptr(i)))
		charRhs := *(*byte)(unsafe.Pointer(uintptr(rhs.array) + uintptr(i)))
		if charLhs != charRhs {
			return false
		}
	}
	return true
}

type stringIterator struct {
	str   stringDescriptor
	index int
}

//go:export stringRange runtime.stringRange
func stringRange(it *stringIterator) (bool, int, rune) {
	if it.index >= it.str.len {
		return false, 0, 0
	} else {
		// Get the current position for the iterator. This will be returned.
		i := it.index

		// Get the rune value at the index in the string's backing array
		// TODO: Support unicode characters
		val := *(*rune)(unsafe.Add(it.str.array, i))

		// Advance the position for the next iteration and then return
		it.index++
		return true, i, val
	}
}
