package volatile

import (
	"unsafe"
)

func LoadInt8(addr *int8) (val int8)
func LoadInt16(addr *int16) (val int16)
func LoadInt32(addr *int32) (val int32)
func LoadInt64(addr *int64) (val int64)

func LoadUint8(addr *uint8) (val uint8)
func LoadUint16(addr *uint16) (val uint16)
func LoadUint32(addr *uint32) (val uint32)
func LoadUint64(addr *uint64) (val uint64)
func LoadUintptr(addr *uintptr) (val uintptr)
func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)

func StoreInt8(addr *int8, val int8)
func StoreInt16(addr *int16, val int16)
func StoreInt32(addr *int32, val int32)
func StoreInt64(addr *int64, val int64)

func StoreUint8(addr *uint8, val uint8)
func StoreUint16(addr *uint16, val uint16)
func StoreUint32(addr *uint32, val uint32)
func StoreUint64(addr *uint64, val uint64)
func StoreUintptr(addr *uintptr, val uintptr)
func StorePointer(addr *unsafe.Pointer, val unsafe.Pointer)
