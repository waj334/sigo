package runtime

import "unsafe"

//sigo:extern abort runtime.abort
func abort()

//sigo:extern exec runtime.exec
func exec(args, fn unsafe.Pointer)

func nilCheck(ptr unsafe.Pointer) {
	if ptr == nil {
		panic("runtime error: invalid memory address or nil pointer dereference")
	}
}
