package _go

import (
	"unsafe"
)

//go:linkname runtime.alloc
func alloc(size uintptr) unsafe.Pointer
