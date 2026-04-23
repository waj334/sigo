package reflectlite

import "unsafe"

// _interface matches the runtime's interface representation.
// Field order must match runtime._interface exactly: {value, valueT}.
type _interface struct {
	value  unsafe.Pointer
	valueT *_type
}

// _slice matches the runtime's slice header layout.
type _slice struct {
	array unsafe.Pointer
	len   int
	cap   int
}

// _string matches the runtime's string header layout.
type _string struct {
	data unsafe.Pointer
	len  int
}
