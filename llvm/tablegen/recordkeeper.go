package tablegen

// #include "tablegen.h"
import "C"

import (
	"runtime"
)

type RecordKeeper interface {
	Ptr() *C.LLVMRecordKeeper
	GetInputFilename() string
	GetClasses() map[string]Record
	GetDefs() map[string]Record
	GetGlobals() map[string]Init
}

type recordKeeper struct {
	ptr *C.LLVMRecordKeeper
}

func NewRecordKeeper() RecordKeeper {
	rk := new(recordKeeper)
	rk.ptr = C.LLVMCreateRecordKeeper()
	runtime.SetFinalizer(rk, func(r *recordKeeper) {
		if r.ptr != nil {
			C.LLVMDisposeRecordKeeper(r.ptr)
			r.ptr = nil
		}
	})
	return rk
}

func (r *recordKeeper) Ptr() *C.LLVMRecordKeeper {
	return r.ptr
}

func (r *recordKeeper) GetInputFilename() string {
	cstr := C.LLVMRecordKeeperGetInputFilename(r.ptr)
	defer C.LLVMDisposeCString(cstr)
	return C.GoString(cstr)
}

func (r *recordKeeper) GetClasses() map[string]Record {
	rm := newRecordMap(C.LLVMRecordKeeperGetClasses(r.ptr))
	m := make(map[string]Record)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}

func (r *recordKeeper) GetDefs() map[string]Record {
	rm := newRecordMap(C.LLVMRecordKeeperGetDefs(r.ptr))
	m := make(map[string]Record)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}

func (r *recordKeeper) GetGlobals() map[string]Init {
	rm := newGlobalMap(C.LLVMRecordKeeperGetGlobals(r.ptr))
	m := make(map[string]Init)
	it := rm.Begin()
	for it.Next() {
		m[it.Key()] = it.Value()
	}
	return m
}
