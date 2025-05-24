package tablegen

// #include "tablegen.h"
import "C"
import "runtime"

type GlobalMap struct {
	ptr *C.LLVMGlobalMap
}

func newGlobalMap(val *C.LLVMGlobalMap) *GlobalMap {
	g := &GlobalMap{ptr: val}
	runtime.SetFinalizer(g, func(g *GlobalMap) {
		if g.ptr != nil {
			C.LLVMDisposeGlobalMap(g.ptr)
			g.ptr = nil
		}
	})
	return g
}

func (g *GlobalMap) Begin() *GlobalMapIterator {
	return newGlobalMapIterator(C.LLVMGlobalMapBegin(g.ptr))
}
