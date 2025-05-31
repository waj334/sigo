package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type GlobalMapIterator struct {
	ptr *C.LLVMGlobalMapIterator
}

func newGlobalMapIterator(val *C.LLVMGlobalMapIterator) *GlobalMapIterator {
	itval := &GlobalMapIterator{ptr: val}
	runtime.SetFinalizer(itval, func(it *GlobalMapIterator) {
		if it.ptr != nil {
			C.LLVMDisposeGlobalMapIterator(it.ptr)
			it.ptr = nil
		}
	})
	return itval
}

func (g *GlobalMapIterator) Key() string {
	cstr := C.LLVMGlobalMapIteratorKey(g.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (g *GlobalMapIterator) Value() Init {
	return newInit(C.LLVMGlobalMapIteratorValue(g.ptr))
}

func (g *GlobalMapIterator) Next() bool {
	return bool(C.LLVMGlobalMapIteratorNext(g.ptr))
}
