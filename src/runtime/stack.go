package runtime

import "unsafe"

//sigo:extern currentStack _current_stack
func currentStack() unsafe.Pointer
