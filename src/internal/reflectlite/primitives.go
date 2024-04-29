package reflectlite

import "unsafe"

type _interface struct {
	typePtr  *_type
	valuePtr unsafe.Pointer
}
