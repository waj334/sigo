package tablegen

// #include "tablegen.h"
import "C"

import (
	"unsafe"
)

type Record struct {
	ptr    *C.LLVMRecord
	parent *RecordKeeper
}

func (r *Record) GetName() string {
	cstr := C.LLVMRecordGetName(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *Record) GetValueAsString(name string) string {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	cstr := C.LLVMRecordGetValueAsString(r.ptr, cname)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *Record) GetValueAsDef(name string) *Record {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	record := C.LLVMRecordGetValueAsDef(r.ptr, cname)
	return r.parent.adoptRecord(record)
}

func (r *Record) GetValueAsOptionalDef(name string) *Record {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	result := C.LLVMRecordGetValueAsOptionalDef(r.ptr, cname)
	if bool(C.LLVMRecordIsNull(result)) {
		return nil
	}
	return r.parent.adoptRecord(result)
}

func (r *Record) GetValueAsBit(name string) bool {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return bool(C.LLVMRecordGetValueAsBit(r.ptr, cname))
}

func (r *Record) GetValueAsBitOrUnset(name string) (bool, bool) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	unset := false
	result := bool(C.LLVMRecordGetValueAsBitOrUnset(r.ptr, cname, (*C.bool)(unsafe.Pointer(&unset))))
	return result, unset
}

func (r *Record) GetValueAsInt(name string) int64 {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return int64(C.LLVMRecordGetValueAsInt(r.ptr, cname))
}

func (r *Record) GetValueAsListOfDefs(name string) []*Record {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	list := C.LLVMRecordGetValueAsListOfDefs(r.ptr, cname)
	defer C.LLVMDisposeRecordList(list)

	sz := int(C.LLVMRecordListSize(list))
	records := make([]*Record, sz)
	for i := range sz {
		record := C.LLVMRecordListValue(list, C.int(i))
		records[i] = r.parent.adoptRecord(record)
	}
	return records
}
