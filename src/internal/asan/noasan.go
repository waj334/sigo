package asan

import (
	"unsafe"
)

const Enabled = false

func Read(addr unsafe.Pointer, len uintptr)  {}
func Write(addr unsafe.Pointer, len uintptr) {}
