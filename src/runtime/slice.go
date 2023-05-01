package runtime

import (
	"unsafe"

	"runtime/internal/allocator"
)

type sliceDescriptor struct {
	array unsafe.Pointer
	len   int
	cap   int
}

//go:export sliceMake runtime.makeSlice
func sliceMake(t unsafe.Pointer, n int, m int) unsafe.Pointer {
	// The length MUST not be greater than the capacity of the slice!
	if n > m {
		// TODO: panic!
		return nil
	}

	elementTypeDesc := (*typeDescriptor)(t)
	result := sliceDescriptor{
		len: n,
		cap: m,
	}

	// Allocate the underlying array
	result.array = allocator.GCAlloc(uintptr(result.cap) * elementTypeDesc.size)

	// Finally, return the new slice
	return unsafe.Pointer(&result.array)
}

//go:export sliceAppend runtime.append
func sliceAppend(base unsafe.Pointer, elems unsafe.Pointer, elementType unsafe.Pointer) unsafe.Pointer {
	baseSlice := (*sliceDescriptor)(base)
	baseLength := uintptr(baseSlice.len)

	elementsSlice := (*sliceDescriptor)(elems)
	elementsLength := uintptr(elementsSlice.len)
	elementTypeDesc := (*typeDescriptor)(elementType)

	result := sliceDescriptor{
		array: baseSlice.array,
		cap:   baseSlice.cap,
	}

	// Check if the elements can fit into the base slice
	result.len = baseSlice.len + elementsSlice.len
	if result.len > baseSlice.cap {
		// Allocate a new underlying array
		result.array = allocator.GCAlloc(uintptr(result.len) * elementTypeDesc.size)

		// Copy the contents of the old array into the new array
		allocator.Memcpy(result.array, baseSlice.array, baseLength*elementTypeDesc.size)

		// Set the capacity to that of the new length
		result.cap = result.len
	}

	// Copy the contents of the elements slice onto the end of the base slice
	allocator.Memcpy(
		unsafe.Add(result.array, baseLength*elementTypeDesc.size),
		elementsSlice.array,
		elementsLength*elementTypeDesc.size)

	// Return the new slice
	return unsafe.Pointer(&result.array)
}

//go:export sliceLen runtime.sliceLen
func sliceLen(s unsafe.Pointer) int {
	slice := (*sliceDescriptor)(s)
	return slice.len
}

//go:export sliceCap runtime.sliceCap
func sliceCap(s unsafe.Pointer) int {
	slice := (*sliceDescriptor)(s)
	return slice.cap
}

//go:export sliceCopy runtime.sliceCopy
func sliceCopy(src, dst, elementType unsafe.Pointer) int {
	srcSlice := (*sliceDescriptor)(src)
	dstSlice := (*sliceDescriptor)(dst)
	elementTypeDesc := (*typeDescriptor)(elementType)

	// Copy either as much as what is available or as much as there is capacity
	// for.
	n := srcSlice.len
	if n > dstSlice.cap {
		n = dstSlice.cap
	}

	// Copy N elements from the src into dst
	allocator.Memcpy(dstSlice.array, srcSlice.array, uintptr(n)*elementTypeDesc.size)

	// Return the number elements copied
	return n
}

//go:export sliceIndex runtime.sliceIndexAddr
func sliceIndexAddr(s unsafe.Pointer, index int, elementType unsafe.Pointer) unsafe.Pointer {
	slice := (*sliceDescriptor)(s)
	elementTypeDesc := (*typeDescriptor)(elementType)
	// Index MUST not be greater than the length of the slice
	if index >= slice.len {
		// TODO: Panic
		return nil
	}
	// Return the address of the element at the specified index
	return unsafe.Add(slice.array, uintptr(index)*elementTypeDesc.size)
}

//go:export sliceAddr runtime.sliceAddr
func sliceAddr(ptr unsafe.Pointer, length, capacity, elementSize, low, high, max int) unsafe.Pointer {
	arr := ptr
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
		// TODO: Panic
		return nil
	}

	slice := sliceDescriptor{
		array: unsafe.Add(arr, newLow*elementSize),
		len:   newHigh - newLow,
		cap:   newMax - newLow,
	}
	return unsafe.Pointer(&slice.array)
}
