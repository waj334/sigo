package tablegen

// #include "tablegen.h"
// #include <stdlib.h>
import "C"

import (
	"runtime"
	"unsafe"
)

type RecordKeeper struct {
	ptr         *C.LLVMRecordKeeper
	liveRecords map[*C.LLVMRecord]*Record
}

func NewRecordKeeper() *RecordKeeper {
	rk := &RecordKeeper{
		ptr:         C.LLVMCreateRecordKeeper(),
		liveRecords: make(map[*C.LLVMRecord]*Record),
	}
	runtime.SetFinalizer(rk, func(r *RecordKeeper) {
		if r.ptr != nil {
			C.LLVMDisposeRecordKeeper(r.ptr)
			r.ptr = nil
		}
	})
	return rk
}

func (rk *RecordKeeper) adoptRecord(cptr *C.LLVMRecord) *Record {
	if cptr == nil {
		return nil
	}
	if r, ok := rk.liveRecords[cptr]; ok {
		return r
	}
	rec := &Record{ptr: cptr, parent: rk}
	runtime.SetFinalizer(rec, func(r *Record) {
		if r.ptr != nil {
			C.LLVMDisposeRecord(r.ptr)
			r.ptr = nil
		}
	})
	rk.liveRecords[cptr] = rec
	return rec
}

func (rk *RecordKeeper) ReleaseAllRecords() {
	for _, r := range rk.liveRecords {
		C.LLVMDisposeRecord(r.ptr)
		r.ptr = nil
	}
	rk.liveRecords = nil
}

func (r *RecordKeeper) Ptr() *C.LLVMRecordKeeper {
	return r.ptr
}

func (r *RecordKeeper) GetInputFilename() string {
	cstr := C.LLVMRecordKeeperGetInputFilename(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *RecordKeeper) GetClasses() map[string]*Record {
	rm := newRecordMap(C.LLVMRecordKeeperGetClasses(r.ptr), r)
	m := make(map[string]*Record)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}

func (r *RecordKeeper) GetDefs() map[string]*Record {
	rm := newRecordMap(C.LLVMRecordKeeperGetDefs(r.ptr), r)
	m := make(map[string]*Record)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}

func (r *RecordKeeper) GetGlobals() map[string]Init {
	rm := newGlobalMap(C.LLVMRecordKeeperGetGlobals(r.ptr))
	m := make(map[string]Init)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}

func (r *RecordKeeper) GetDerivedRecords(className string) []*Record {
	cname := C.CString(className)
	defer C.free(unsafe.Pointer(cname))

	list := C.LLVMRecordKeeperGetAllDerivedDefinitions(r.ptr, cname)
	defer C.LLVMDisposeRecordList(list)

	sz := int(C.LLVMRecordListSize(list))
	records := make([]*Record, sz)
	for i := range sz {
		records[i] = r.adoptRecord(C.LLVMRecordListValue(list, C.int(i)))
	}
	return records
}
