package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type Record interface {
	GetName() string
}

type record struct {
	ptr *C.LLVMRecord
}

func newRecord(val *C.LLVMRecord) Record {
	rval := new(record)
	rval.ptr = val
	runtime.SetFinalizer(rval, func(r *record) {
		if r.ptr != nil {
			C.LLVMDisposeRecord(r.ptr)
			r.ptr = nil
		}
	})
	return rval
}

func (r *record) GetName() string {
	cstr := C.LLVMRecordGetName(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}
