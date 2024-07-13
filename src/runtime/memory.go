package runtime

import (
	"unsafe"
)

//sigo:extern malloc malloc
func malloc(size uintptr) unsafe.Pointer

//sigo:extern realloc realloc
func realloc(ptr unsafe.Pointer, size uintptr) unsafe.Pointer

//sigo:extern free free
func free(ptr unsafe.Pointer)

//sigo:extern memcpy memcpy
func memcpy(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer

//sigo:extern memmove memmove
func memmove(dst, src unsafe.Pointer, num uintptr) unsafe.Pointer

//sigo:extern memset memset
func memset(ptr unsafe.Pointer, value int, num uintptr) unsafe.Pointer

//sigo:extern memcmp memcmp
func memcmp(dst, src unsafe.Pointer, num uintptr) int
