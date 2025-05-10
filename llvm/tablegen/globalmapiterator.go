package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type GlobalMapIterator interface {
	Key() string
	Value() Init
	Next() bool
}

type globalMapIterator struct {
	ptr *C.LLVMGlobalMapIterator
}

func newGlobalMapIterator(val *C.LLVMGlobalMapIterator) GlobalMapIterator {
	itval := new(globalMapIterator)
	itval.ptr = val
	runtime.SetFinalizer(itval, func(it *globalMapIterator) {
		if it.ptr != nil {
			C.LLVMDisposeGlobalMapIterator(it.ptr)
			it.ptr = nil
		}
	})
	return itval
}

func (g *globalMapIterator) Key() string {
	cstr := C.LLVMGlobalMapIteratorKey(g.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (g *globalMapIterator) Value() Init {
	return newInit(C.LLVMGlobalMapIteratorValue(g.ptr))
}

func (g *globalMapIterator) Next() bool {
	return bool(C.LLVMGlobalMapIteratorNext(g.ptr))
}
