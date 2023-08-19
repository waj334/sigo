package reflectlite

import "unsafe"

const (
	fnvBasis uint64 = 0xcbf29ce484222325
	fnvPrime uint64 = 0x100000001b3
)

func computeFnv(ptr unsafe.Pointer, size uintptr) uint64 {
	hash := fnvBasis
	for i := uintptr(0); i < size; i++ {
		b := *(*byte)(unsafe.Add(ptr, i))
		hash = hash * fnvPrime
		hash = hash ^ uint64(b)
	}
	return hash
}
