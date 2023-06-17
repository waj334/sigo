package atsamx51

import (
	"runtime/arm/cortexm/sam/chip"
	"unsafe"
	"volatile"
)

type Interrupt int16

func (i Interrupt) EnableIRQ() {
	chip.NVIC.ISER[i>>5].SetSETENA(1 << (i & 0x1F))
}

func (i Interrupt) DisableIRQ() {
	chip.NVIC.ICER[i>>5].SetCLRENA(1 << (i & 0x1F))
}

func (i Interrupt) SetPriority(priority uint8) {
	//index := uint32(i) / 4
	//shift := uint32(i%4) * 8
	ip := (*uint8)(unsafe.Add(unsafe.Pointer(&chip.NVIC.IP[0]), i))
	volatile.StoreUint8(ip, priority)
	//ip.SetPRI0((ip.GetPRI0() &^ (0xFF << shift)) | (uint32(priority&0xFF) << shift))
}
