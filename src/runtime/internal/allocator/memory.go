package allocator

import (
	"unsafe"
)

//go:linkname alloc runtime.alloc
func alloc(size uintptr) unsafe.Pointer

//sigo:extern malloc malloc
func malloc(size uintptr) unsafe.Pointer

//sigo:extern free free
func free(ptr unsafe.Pointer)

func Malloc(size uintptr) unsafe.Pointer {
	return malloc(size)
}

func Free(ptr unsafe.Pointer) {
	free(ptr)
}
