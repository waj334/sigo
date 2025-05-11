package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
	"unsafe"
)

type Record interface {
	GetName() string
	GetValueAsString(name string) string
	GetValueAsDef(name string) Record
	GetValueAsOptionalDef(name string) Record
	GetValueAsBit(name string) bool
	GetValueAsBitOrUnset(name string) (bool, bool)
	GetValueAsInt(name string) int64
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

func (r *record) GetValueAsString(name string) string {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	cstr := C.LLVMRecordGetValueAsString(r.ptr, cname)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *record) GetValueAsDef(name string) Record {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return newRecord(C.LLVMRecordGetValueAsDef(r.ptr, cname))
}

func (r *record) GetValueAsOptionalDef(name string) Record {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	result := C.LLVMRecordGetValueAsOptionalDef(r.ptr, cname)
	if bool(C.LLVMRecordIsNull(result)) {
		return nil
	}
	return newRecord(result)
}

func (r *record) GetValueAsBit(name string) bool {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return bool(C.LLVMRecordGetValueAsBit(r.ptr, cname))
}

func (r *record) GetValueAsBitOrUnset(name string) (bool, bool) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	unset := false
	result := bool(C.LLVMRecordGetValueAsBitOrUnset(r.ptr, cname, (*C.bool)(unsafe.Pointer(&unset))))
	return result, unset
}

func (r *record) GetValueAsInt(name string) int64 {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return int64(C.LLVMRecordGetValueAsInt(r.ptr, cname))
}
