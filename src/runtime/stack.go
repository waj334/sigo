package runtime

import "unsafe"

//sigo:extern currentStack runtime.currentStack
func currentStack() unsafe.Pointer
