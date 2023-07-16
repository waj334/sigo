package cortexm

import (
	"unsafe"
	"volatile"
)

var (
	SYST = (*SysTick)(unsafe.Pointer(uintptr(0xE000E010)))
)

type SysTick struct {
	CSR   SYST_CSR
	RVR   SYST_RVR
	CVR   SYST_CVR
	CALIB SYST_CALIB
}

type SYST_CSR uint32

func (reg *SYST_CSR) SetENABLE(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1
	} else {
		value &= ^(value & 0x1)
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SYST_CSR) GetENABLE() bool {
	return volatile.LoadUint32((*uint32)(reg))&0x1 != 0
}

func (reg *SYST_CSR) SetTICKINT(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 1
	} else {
		value &= ^(value & (0x1 << 1))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SYST_CSR) GetTICKINIT() bool {
	return volatile.LoadUint32((*uint32)(reg))&(0x1<<1) != 0
}

func (reg *SYST_CSR) SetCLKSOURCE(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 2
	} else {
		value &= ^(value & (0x1 << 2))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SYST_CSR) GetCLKSOURCE() bool {
	return volatile.LoadUint32((*uint32)(reg))&(0x1<<2) != 0
}

func (reg *SYST_CSR) GetCOUNTFLAG() bool {
	return volatile.LoadUint32((*uint32)(reg))&(0x1<<16) != 0
}

type SYST_RVR uint32

func (reg *SYST_RVR) SetRELOAD(value uint32) {
	volatile.StoreUint32((*uint32)(reg), value&0xFFFFFF)
}

func (reg *SYST_RVR) GetRELOAD() uint32 {
	return volatile.LoadUint32((*uint32)(reg))
}

type SYST_CVR uint32

func (reg *SYST_CVR) SetVALUE(value uint32) {
	// Any write to the register clears the register to zero
	volatile.StoreUint32((*uint32)(reg), 1)
}

func (reg *SYST_CVR) GetVALUE() uint32 {
	return volatile.LoadUint32((*uint32)(reg))
}

type SYST_CALIB uint32

func (reg *SYST_CALIB) GetTENMS() uint32 {
	return volatile.LoadUint32((*uint32)(reg)) & 0xFFFFFF
}

func (reg *SYST_CALIB) GetSKEW() bool {
	return volatile.LoadUint32((*uint32)(reg))&(0x1<<30) != 0
}

func (reg *SYST_CALIB) GetNOREF() bool {
	return volatile.LoadUint32((*uint32)(reg))&(0x1<<31) != 0
}
