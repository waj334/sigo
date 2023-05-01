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

//sigo:extern memcpy memcpy
func memcpy(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer

//sigo:extern memmove memmove
func memmove(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer

//sigo:extern memset memset
func memset(ptr unsafe.Pointer, value int, num uintptr) unsafe.Pointer

func GCAlloc(size uintptr) unsafe.Pointer {
	return alloc(size)
}

func Malloc(size uintptr) unsafe.Pointer {
	return malloc(size)
}

func Free(ptr unsafe.Pointer) {
	free(ptr)
}

func Memcpy(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer {
	return memcpy(dst, src, num)
}

func Memmove(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer {
	return memmove(dst, src, num)
}

func Memset(ptr unsafe.Pointer, value int, num uintptr) unsafe.Pointer {
	return memset(ptr, value, num)
}
