package runtime

import (
	"unsafe"
)

type deferEntry struct {
	fn      unsafe.Pointer
	args    unsafe.Pointer
	numArgs int
	next    *unsafe.Pointer
}
