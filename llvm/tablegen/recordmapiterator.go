package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type RecordMapIterator struct {
	ptr    *C.LLVMRecordMapIterator
	parent *RecordMap
}

func newRecordMapIterator(val *C.LLVMRecordMapIterator, parent *RecordMap) *RecordMapIterator {
	itval := &RecordMapIterator{
		ptr:    val,
		parent: parent,
	}
	runtime.SetFinalizer(itval, func(it *RecordMapIterator) {
		if it.ptr != nil {
			C.LLVMDisposeRecordMapIterator(it.ptr)
			it.ptr = nil
		}
	})
	return itval
}

func (r *RecordMapIterator) Key() string {
	cstr := C.LLVMRecordMapIteratorKey(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *RecordMapIterator) Value() *Record {
	return r.parent.parent.adoptRecord(C.LLVMRecordMapIteratorValue(r.ptr))
}

func (r *RecordMapIterator) Next() bool {
	return bool(C.LLVMRecordMapIteratorNext(r.ptr))
}
