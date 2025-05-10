package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type Init interface {
}

type initV struct {
	ptr *C.LLVMInit
}

func newInit(val *C.LLVMInit) Init {
	ival := new(initV)
	ival.ptr = val
	runtime.SetFinalizer(ival, func(i *initV) {
		if i.ptr != nil {
			C.LLVMDisposeInit(i.ptr)
			i.ptr = nil
		}
	})
	return ival
}
