package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type RecordMap struct {
	ptr    *C.LLVMRecordMap
	parent *RecordKeeper
}

func newRecordMap(val *C.LLVMRecordMap, parent *RecordKeeper) *RecordMap {
	rval := &RecordMap{
		ptr:    val,
		parent: parent,
	}
	runtime.SetFinalizer(rval, func(r *RecordMap) {
		if r.ptr != nil {
			C.LLVMDisposeRecordMap(r.ptr)
			r.ptr = nil
		}
	})
	return rval
}

func (r *RecordMap) Begin() *RecordMapIterator {
	return newRecordMapIterator(C.LLVMRecordMapBegin(r.ptr), r)
}
