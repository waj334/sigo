package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type RecordMap interface {
	Begin() RecordMapIterator
}

type recordMap struct {
	ptr *C.LLVMRecordMap
}

func newRecordMap(val *C.LLVMRecordMap) RecordMap {
	rval := new(recordMap)
	rval.ptr = val
	runtime.SetFinalizer(rval, func(r *recordMap) {
		if r.ptr != nil {
			C.LLVMDisposeRecordMap(r.ptr)
			r.ptr = nil
		}
	})
	return rval
}

func (r *recordMap) Begin() RecordMapIterator {
	return newRecordMapIterator(C.LLVMRecordMapBegin(r.ptr))
}
