package runtime

import "unsafe"

type sliceDescriptor struct {
	array unsafe.Pointer
	len   int
	cap   int
}

//go:export sliceMake runtime.sliceMake
func sliceMake(typ unsafe.Pointer, elements unsafe.Pointer, count int) unsafe.Pointer {
	return unsafe.Pointer(&sliceDescriptor{})
}

//go:export sliceAppend runtime.Append
func sliceAppend(buf unsafe.Pointer, elems unsafe.Pointer) unsafe.Pointer {
	return nil
}

//go:export sliceLen runtime.sliceLen
func sliceLen(s unsafe.Pointer) int {
	return 0
}

//go:export sliceCap runtime.sliceCap
func sliceCap(s unsafe.Pointer) int {
	return 0
}

//go:export sliceCopy runtime.sliceCopy
func sliceCopy(dst unsafe.Pointer) int {
	return 0
}

//go:export sliceIndex runtime.sliceIndex
func sliceIndex(s unsafe.Pointer, index int, elementSize int64) unsafe.Pointer {
	return nil
}

//go:export sliceAddr runtime.sliceAddr
func sliceAddr(ptr unsafe.Pointer, length, capacity, elementSize, low, high, max int) unsafe.Pointer {
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
			return nil
		}
	}

	if low >= 0 {
		if low < 0 || low > length || low > newCap {
			// TODO: Panic
			return nil
		}
		newLow = low
	}

	if high >= 0 {
		if high < 0 || high > length || high > newCap {
			// TODO: Panic
			return nil
		}
		newHigh = high
	}

	if newCap > length {
		// TODO: Allocate a new underlying array for the slice
		// TODO: Copy old array into the new array
		return nil
	} else {
		end := newHigh - newLow
		if end > newCap {
			newHigh = low + newCap
		}
	}

	return unsafe.Pointer(&sliceDescriptor{
		array: unsafe.Add(arr, newLow),
		len:   newHigh - newLow,
		cap:   newCap,
	})
}
