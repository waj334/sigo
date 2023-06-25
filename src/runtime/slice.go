package runtime

import (
	"unsafe"
)

type _slice struct {
	array unsafe.Pointer
	len   int
	cap   int
}

func sliceMake(t unsafe.Pointer, n int, m int) _slice {
	// The length MUST not be greater than the capacity of the slice!
	if n > m {
		panic("runtime error: makeslice: cap out of range")
	}

	elementTypeDesc := (*_type)(t)
	result := _slice{
		len: n,
		cap: m,
	}

	// Allocate the underlying array
	result.array = alloc(uintptr(result.cap) * elementTypeDesc.size)

	// Finally, return the new slice
	return result
}

func sliceAppend(base _slice, incoming _slice, elementType *_type) _slice {
	newLen := base.len + incoming.len
	newCap := base.cap
	array := base.array
	offset := uintptr(base.len) * elementType.size
	if newCap < newLen {
		newCap = newLen

		// Allocate a new underlying array
		array = alloc(uintptr(newCap) * elementType.size)

		// Copy the contents of the old array into the new array
		memcpy(array, base.array, offset)
	}

	// Copy the incoming elements after the existing elements
	memcpy(unsafe.Add(array, offset), incoming.array, uintptr(incoming.len)*elementType.size)

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

func sliceCopy(src, dst _slice, elementType *_type) int {
	// Copy either as much as what is available or as much as there is capacity
	// for.
	n := src.len
	if n > dst.cap {
		n = dst.cap
	}

	// Copy N elements from the src into dst
	memcpy(dst.array, src.array, uintptr(n)*elementType.size)

	// Return the number elements copied
	return n
}

func sliceIndexAddr(s _slice, index int, elementType *_type) unsafe.Pointer {
	// Index MUST not be greater than the length of the slice
	if index >= s.len {
		panic("runtime: index out of range")
	}
	// Return the address of the element at the specified index
	return unsafe.Add(s.array, uintptr(index)*elementType.size)
}

func sliceAddr(ptr unsafe.Pointer, length, capacity, elementSize, low, high, max int) _slice {
	newLow := 0
	newHigh := length
	newMax := capacity

	if low >= 0 {
		newLow = low
	}

	if high >= 0 {
		newHigh = high
	}

	if max >= 0 {
		newMax = max
	}

	if newLow < 0 || newHigh < newLow || newMax < newHigh || newHigh > length || newMax > capacity {
		panic("runtime error: slice bounds out of range [TODO:TODO:TODO]")
	}

	return _slice{
		array: unsafe.Add(ptr, newLow*elementSize),
		len:   newHigh - newLow,
		cap:   newMax - newLow,
	}
}

func sliceString(str _string) _slice {
	result := _slice{
		array: *(*unsafe.Pointer)(alloc(uintptr(str.len))),
		len:   str.len,
		cap:   str.len,
	}
	memcpy(result.array, str.array, uintptr(str.len))
	return result
}
