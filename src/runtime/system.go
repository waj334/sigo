package runtime

import "unsafe"

//sigo:extern abort _abort
func abort()

//go:export nilCheck runtime.nilCheck
func nilCheck(ptr unsafe.Pointer) {
	if ptr == nil {
		panic("runtime error: invalid memory address or nil pointer dereference")
	}
}

// initPackages executes the init functions implicitly defines by each
// package imported by the program. This function is internally defined by the compiler.
//
//go:export initPackages runtime.initPackages
//func initPackages()
