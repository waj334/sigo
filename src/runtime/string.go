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

func stringLen(descriptor _string) int {
	str := descriptor
	return str.len
}

func stringConcat(lhs _string, rhs _string) _string {
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

func stringIndexAddr(str _string, index int) unsafe.Pointer {
	// Index MUST not be greater than the length of the string
	if index >= str.len {
		panic("runtime: index out of range")
	}
	// Return the address of the element at the specified index
	return unsafe.Add(str.array, uintptr(index))
}

func stringCompare(lhs, rhs _string) bool {
	// Fast path
	if lhs.len != rhs.len {
		return false
	}
	return strncmp(lhs.array, rhs.array, uintptr(lhs.len)) == 0
}

type _stringIterator struct {
	str   _string
	index int
}

func stringRange(it *_stringIterator) (bool, int, rune) {
	if it.index >= it.str.len {
		return false, 0, 0
	} else {
		// Get the current position for the iterator. This will be returned.
		i := it.index

		// TODO: Support unicode characters
		// Perform an 4-byte aligned read
		val := *(*rune)(unsafe.Add(it.str.array, 4*(i/4)))

		// Get the rune value at the index in the string's backing array
		// Shift to access the intended byte
		val = (val >> (8 * (it.index % 4))) & 0xFF

		// Advance the position for the next iteration and then return
		it.index++
		return true, i, val
	}
}

func stringSlice(str _string, low, high int) _string {
	if low == -1 {
		low = 0
	}

	if high == -1 {
		high = str.len
	}

	if 0 > low || low > high || low > str.len || high > str.len {
		panic("runtime error: slice out of bounds [::]")
	}

	newLen := high - low

	result := _string{
		array: alloc(uintptr(newLen)),
		len:   newLen,
	}
	memcpy(result.array, unsafe.Add(str.array, low), uintptr(newLen))
	return result
}

func stringToSlice(str _string) _slice {
	result := _slice{
		array: alloc(uintptr(str.len)),
		len:   str.len,
		cap:   str.len,
	}
	memcpy(result.array, str.array, uintptr(str.len))
	return result
}

func sliceToString(s _slice) _string {
	result := _string{
		array: alloc(uintptr(s.len)),
		len:   s.len,
	}
	memcpy(result.array, s.array, uintptr(s.len))
	return result
}
