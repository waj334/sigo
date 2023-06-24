package runtime

import "unsafe"

//sigo:extern abort runtime.abort
func abort()

func nilCheck(ptr unsafe.Pointer) {
	if ptr == nil {
		panic("runtime error: invalid memory address or nil pointer dereference")
	}
}
