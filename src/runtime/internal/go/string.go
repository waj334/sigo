package _go

import "unsafe"

type stringDescriptor struct {
	array unsafe.Pointer
	len   int
}

//go:linkname runtime.stringLen
func stringLen(descriptor stringDescriptor) int {
	return 0
}
