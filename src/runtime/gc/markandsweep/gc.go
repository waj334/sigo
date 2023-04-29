package markandsweep

import "unsafe"
import "runtime/internal/allocator"

//go:export alloc runtime.alloc
func alloc(size uintptr) unsafe.Pointer {
	addr := allocator.Malloc(size)
	return addr
}
