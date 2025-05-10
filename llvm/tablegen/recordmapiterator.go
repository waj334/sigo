package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type RecordMapIterator interface {
	Key() string
	Value() Record
	Next() bool
}

type recordMapIterator struct {
	ptr *C.LLVMRecordMapIterator
}

func newRecordMapIterator(val *C.LLVMRecordMapIterator) RecordMapIterator {
	itval := new(recordMapIterator)
	itval.ptr = val
	runtime.SetFinalizer(itval, func(it *recordMapIterator) {
		if it.ptr != nil {
			C.LLVMDisposeRecordMapIterator(it.ptr)
			it.ptr = nil
		}
	})
	return itval
}

func (r *recordMapIterator) Key() string {
	cstr := C.LLVMRecordMapIteratorKey(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *recordMapIterator) Value() Record {
	return newRecord(C.LLVMRecordMapIteratorValue(r.ptr))
}

func (r *recordMapIterator) Next() bool {
	return bool(C.LLVMRecordMapIteratorNext(r.ptr))
}
