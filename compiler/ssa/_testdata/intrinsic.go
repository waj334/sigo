// RUN: FileCheck %s

package main

import (
	"unsafe"
	"volatile"
)

func volatileIntrinsics() {
	var i8 int8
	var i16 int16
	var i32 int32
	var i64 int64

	var ui8 uint8
	var ui16 uint16
	var ui32 uint32
	var ui64 uint64
	var uptr uintptr
	var ptr unsafe.Pointer

	i8 = volatile.LoadInt8(&i8)
	i16 = volatile.LoadInt16(&i16)
	i32 = volatile.LoadInt32(&i32)
	i64 = volatile.LoadInt64(&i64)

	ui8 = volatile.LoadUint8(&ui8)
	ui16 = volatile.LoadUint16(&ui16)
	ui32 = volatile.LoadUint32(&ui32)
	ui64 = volatile.LoadUint64(&ui64)
	uptr = volatile.LoadUintptr(&uptr)
	ptr = volatile.LoadPointer(&ptr)

	volatile.StoreInt8(&i8, 0)
	volatile.StoreInt16(&i16, 0)
	volatile.StoreInt32(&i32, 0)
	volatile.StoreInt64(&i64, 0)

	volatile.StoreUint8(&ui8, 0)
	volatile.StoreUint16(&ui16, 0)
	volatile.StoreUint32(&ui32, 0)
	volatile.StoreUint64(&ui64, 0)
	volatile.StoreUintptr(&uptr, 0)
	volatile.StorePointer(&ptr, nil)
}
