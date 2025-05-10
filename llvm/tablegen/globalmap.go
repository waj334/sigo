package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type GlobalMap interface {
	Begin() GlobalMapIterator
}

type globalMap struct {
	ptr *C.LLVMGlobalMap
}

func newGlobalMap(val *C.LLVMGlobalMap) GlobalMap {
	gval := new(globalMap)
	gval.ptr = val
	runtime.SetFinalizer(gval, func(g *globalMap) {
		if g.ptr != nil {
			C.LLVMDisposeGlobalMap(g.ptr)
			g.ptr = nil
		}
	})
	return gval
}

func (g *globalMap) Begin() GlobalMapIterator {
	return newGlobalMapIterator(C.LLVMGlobalMapBegin(g.ptr))
}
