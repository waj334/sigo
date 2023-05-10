package runtime

import "unsafe"

//go:linkname currentStack _current_stack
func currentStack() unsafe.Pointer
