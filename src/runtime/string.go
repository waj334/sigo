package runtime

import (
	"unsafe"
)

type _string struct {
	array unsafe.Pointer
	len   int
}

//sigo:extern strncmp strncmp

func strncmp(str1, str2 unsafe.Pointer, size uintptr) int

func stringLen(descriptor *_string) int {
	str := descriptor
	return str.len
}

func stringConcat(lhs *_string, rhs *_string) _string {
	// Allocate storage buffer for the new string
	newLen := lhs.len + rhs.len
	newArray := alloc(uintptr(newLen))

	// Copy the contents of the strings into the new buffer
	memcpy(newArray, lhs.array, uintptr(lhs.len))
	memcpy(unsafe.Add(newArray, lhs.len), rhs.array, uintptr(rhs.len))

	// Return a new string
	return _string{
		array: newArray,
		len:   newLen,
	}
}

func stringIndexAddr(str *_string, index int) unsafe.Pointer {
	// Index MUST not be greater than the length of the string
	if index >= str.len {
		panic("runtime: index out of range")
	}
	// Return the address of the element at the specified index
	return unsafe.Add(str.array, uintptr(index))
}

func stringCompare(lhs, rhs *_string) bool {
	// Fast path
	if lhs.len != rhs.len {
		return false
	}
	return strncmp(lhs.array, rhs.array, uintptr(lhs.len)) == 0
}

type stringIterator struct {
	str   _string
	index int
}

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
