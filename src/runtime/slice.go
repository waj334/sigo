package runtime

import (
	"unsafe"
)

type _slice struct {
	array unsafe.Pointer
	len   int
	cap   int
}

func sliceMake(elementType *_type, n int, m int) _slice {
	// The length MUST not be greater than the capacity of the slice!
	if n > m {
		panic("runtime error: makeslice: cap out of range")
	}

	result := _slice{
		len: n,
		cap: m,
	}

	// Allocate the underlying array
	result.array = alloc(uintptr(result.cap) * uintptr(elementType.size))

	// Finally, return the new slice
	return result
}

func sliceAppend(base _slice, incoming _slice, elementType *_type) _slice {
	newLen := base.len + incoming.len
	newCap := base.cap
	array := base.array
	offset := uintptr(base.len) * uintptr(elementType.size)
	if newCap < newLen {
		newCap = newLen

		// Allocate a new underlying array
		array = alloc(uintptr(newCap) * uintptr(elementType.size))

		// Copy the contents of the old array into the new array
		memcpy(array, base.array, offset)
	}

	// Copy the incoming elements after the existing elements
	memcpy(unsafe.Add(array, offset), incoming.array, uintptr(incoming.len)*uintptr(elementType.size))

	// Return a new slice
	return _slice{
		array: array,
		len:   newLen,
		cap:   newCap,
	}
}

func sliceLen(s _slice) int {
	return s.len
}

func sliceCap(s _slice) int {
	return s.cap
}

func sliceClear(s _slice, elementType *_type) {
	// Zero out the backing array of the slice.
	memset(s.array, 0, uintptr(elementType.size)*uintptr(s.len))
}

func sliceCopy(dst, src _slice, elementType *_type) int {
	// Copy either as much as what is available or as much as there is capacity
	// for.
	n := src.len
	if n > dst.cap {
		n = dst.cap
	}

	// Copy N elements from the src into dst
	memcpy(dst.array, src.array, uintptr(n)*uintptr(elementType.size))

	// Return the number elements copied
	return n
}

func sliceCopyString(dst _slice, src _string) int {
	n := src.len
	if n > dst.cap {
		n = dst.cap
	}

	// Copy N chars from the src into dst
	memcpy(dst.array, src.array, uintptr(n))

	// Return the number elements copied
	return n
}

func sliceIndexAddr(s _slice, index int, elementType *_type) unsafe.Pointer {
	// Index MUST not be greater than the length of the slice
	if index >= s.len {
		panic("runtime: index out of range")
	}
	// Return the address of the element at the specified index
	return unsafe.Add(s.array, uintptr(index)*uintptr(elementType.size))
}

func sliceReslice(s _slice, info *_type, low, high, max int) _slice {
	elementType := (*_type)(info.data)

	if low == -1 {
		low = 0
	}

	if high == -1 {
		high = s.len
	}

	if max == -1 {
		max = s.cap
	}

	if 0 > low || low > high || low > max ||
		high < low || high > max ||
		max < low || max < high {
		panic("runtime error: slice out of bounds [::]")
	}

	return _slice{
		array: unsafe.Add(s.array, uintptr(low)*uintptr(elementType.size)),
		len:   high - low,
		cap:   max - low,
	}
}

func sliceAddr(ptr unsafe.Pointer, low, high, length int, stride uintptr) _slice {
	if low == -1 {
		low = 0
	}

	if high == -1 {
		high = length
	}

	if 0 > low || low > high || low > length || high > length {
		panic("runtime error: slice out of bounds [::]")
	}

	return _slice{
		array: unsafe.Add(ptr, uintptr(low)*uintptr(stride)),
		len:   high - low,
		cap:   high - low,
	}
}

func sliceIsNil(s _slice) bool {
	return s.array == nil
}

func sliceData(s _slice) unsafe.Pointer {
	return s.array
}

func slice(ptr unsafe.Pointer, len int) _slice {
	return _slice{
		array: ptr,
		len:   len,
		cap:   len,
	}
}
