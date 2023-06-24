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

func sliceAppend(base unsafe.Pointer, elems unsafe.Pointer, elementType unsafe.Pointer) _slice {
	baseSlice := (*_slice)(base)
	baseLength := uintptr(baseSlice.len)

	elementsSlice := (*_slice)(elems)
	elementsLength := uintptr(elementsSlice.len)
	elementTypeDesc := (*_type)(elementType)

	result := _slice{
		array: baseSlice.array,
		cap:   baseSlice.cap,
	}

	// Check if the elements can fit into the base slice
	result.len = baseSlice.len + elementsSlice.len
	if result.len > baseSlice.cap {
		// Allocate a new underlying array
		result.array = *(*unsafe.Pointer)(alloc(uintptr(result.len) * elementTypeDesc.size))

		// Copy the contents of the old array into the new array
		memcpy(result.array, baseSlice.array, baseLength*elementTypeDesc.size)

		// Set the capacity to that of the new length
		result.cap = result.len
	}

	// Copy the contents of the elements slice onto the end of the base slice
	memcpy(
		unsafe.Add(result.array, baseLength*elementTypeDesc.size),
		elementsSlice.array,
		elementsLength*elementTypeDesc.size)

	// Return the new slice
	return result
}

func sliceLen(s unsafe.Pointer) int {
	slice := (*_slice)(s)
	return slice.len
}

func sliceCap(s unsafe.Pointer) int {
	slice := (*_slice)(s)
	return slice.cap
}

func sliceCopy(src, dst, elementType unsafe.Pointer) int {
	srcSlice := (*_slice)(src)
	dstSlice := (*_slice)(dst)
	elementTypeDesc := (*_type)(elementType)

	// Copy either as much as what is available or as much as there is capacity
	// for.
	n := srcSlice.len
	if n > dstSlice.cap {
		n = dstSlice.cap
	}

	// Copy N elements from the src into dst
	memcpy(dstSlice.array, srcSlice.array, uintptr(n)*elementTypeDesc.size)

	// Return the number elements copied
	return n
}

func sliceIndexAddr(s *_slice, index int, elementType *_type) unsafe.Pointer {
	//slice := (*_slice)(s)
	//elementTypeDesc := (*_type)(elementType)
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

func sliceString(s string) _slice {
	str := (*_string)(unsafe.Pointer(&s))
	result := _slice{
		array: *(*unsafe.Pointer)(alloc(uintptr(str.len))),
		len:   str.len,
		cap:   str.len,
	}
	memcpy(result.array, str.array, uintptr(str.len))
	return result
}
