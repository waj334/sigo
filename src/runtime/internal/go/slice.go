package _go

import "unsafe"

type sliceDescriptor struct {
	array unsafe.Pointer
	len   int
	cap   int
}

//go:linkname runtime.sliceMake
func sliceMake(typ *unsafe.Pointer, elements unsafe.Pointer, count int) sliceDescriptor {
	return sliceDescriptor{}
}

//go:linkname runtime.Append
func sliceAppend(buf sliceDescriptor, elems unsafe.Pointer) unsafe.Pointer {
	return nil
}

//go:linkname runtime.sliceLen
func sliceLen(s sliceDescriptor) int {
	return 0
}

//go:linkname runtime.sliceCap
func sliceCap(s sliceDescriptor) int {
	return 0
}

//go:linkname runtime.sliceCopy
func sliceCopy(dst sliceDescriptor) int {
	return 0
}

//go:linkname runtime.sliceIndex
func sliceIndex(s sliceDescriptor, index int, elementSize int64) unsafe.Pointer {
	return nil
}

//go:linkname runtime.sliceAddr
func sliceAddr(ptr unsafe.Pointer, length, capacity, elementSize, low, high, max int) sliceDescriptor {
	arr := ptr
	newLow := 0
	newHigh := length
	newCap := capacity

	if max >= 0 {
		newCap = max
	}

	// TODO: This logic can be optimized
	if low >= 0 && high >= 0 {
		if high < low {
			// TODO: Panic
			return sliceDescriptor{}
		}
	}

	if low >= 0 {
		if low < 0 || low > length || low > newCap {
			// TODO: Panic
			return sliceDescriptor{}
		}
		newLow = low
	}

	if high >= 0 {
		if high < 0 || high > length || high > newCap {
			// TODO: Panic
			return sliceDescriptor{}
		}
		newHigh = high
	}

	if newCap > length {
		// TODO: Allocate a new underlying array for the slice
		// TODO: Copy old array into the new array
		return sliceDescriptor{}
	} else {
		end := newHigh - newLow
		if end > newCap {
			newHigh = low + newCap
		}
	}

	return sliceDescriptor{
		array: unsafe.Add(arr, newLow),
		len:   newHigh - newLow,
		cap:   newCap,
	}
}
