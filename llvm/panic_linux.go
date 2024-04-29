package llvm

import "C"

//export gopanic
func gopanic(cmsg *C.char) {
	msg := C.GoString(cmsg)
	panic(msg)
}
