package reflectlite

import "unsafe"

// Swapper returns a function that swaps the elements in the provided slice.
// Swapper panics if the provided interface is not a slice.
func Swapper(slice any) func(i, j int) {
	v := ValueOf(slice)
	if v.Kind() != Slice {
		panic("reflect.Swapper: provided value is not a slice")
	}

	// Get the slice header and element size.
	sl := (*_slice)(v.ptr)
	elemType := (*_type)(v.typ.data)
	size := elemType.size

	// Capture base pointer and element size for the closure.
	base := sl.array
	return func(i, j int) {
		pi := unsafe.Add(base, uintptr(i)*size)
		pj := unsafe.Add(base, uintptr(j)*size)
		// Swap size bytes between pi and pj.
		for k := uintptr(0); k < size; k++ {
			bi := (*byte)(unsafe.Add(pi, k))
			bj := (*byte)(unsafe.Add(pj, k))
			*bi, *bj = *bj, *bi
		}
	}
}
