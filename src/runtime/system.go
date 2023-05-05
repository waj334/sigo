package runtime

import "unsafe"

//go:linkname abort _abort
func abort()

//go:export nilCheck runtime.nilCheck
func nilCheck(ptr unsafe.Pointer) {
	if ptr == nil {
		panic("runtime error: invalid memory address or nil pointer dereference")
	}
}
