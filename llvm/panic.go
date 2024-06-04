package llvm

/*
#include "panic.h"
*/
import "C"

func init() {
	InitHandler()
}

//export gopanic
func gopanic(cmsg *C.char) {
	msg := C.GoString(cmsg)
	panic(msg)
}

func InitHandler() {
	C.init_panic_handler()
}
