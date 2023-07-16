package cortexm

import (
	"unsafe"
	"volatile"
)

var (
	NVIC = (*NVIC_STR)(unsafe.Pointer(uintptr(0xE000E100)))
)

type (
	NVIC_STR struct {
		ISER [16]uint32
		_    [64]byte
		ICER [16]uint32
		_    [64]byte
		ISPR [16]uint32
		_    [64]byte
		ICPR [16]uint32
		_    [256]byte
		IPRn [124]uint8
	}
)

type Interrupt int16

func (i Interrupt) EnableIRQ() {
	volatile.StoreUint32(&NVIC.ISER[i>>5], 1<<(i&0x1F))
}

func (i Interrupt) DisableIRQ() {
	volatile.StoreUint32(&NVIC.ICER[i>>5], 1<<(i&0x1F))
}

func (i Interrupt) SetPriority(priority uint8) {
	volatile.StoreUint8(&NVIC.IPRn[i], priority)
}
