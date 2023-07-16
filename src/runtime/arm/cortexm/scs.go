package cortexm

import (
	"unsafe"
	"volatile"
)

var (
	SCS = (*SystemControlSpace)(unsafe.Pointer(uintptr(0xE000ED00)))
)

type (
	SystemControlSpace struct {
		CPUID SCS_CPUID
		ICSR  SCS_ICSR
		VTOR  SCS_VTOR
		AIRCR SCS_AIRCR
		SCR   SCS_SCR
		CCR   SCS_CCR
		SHPR1 SCS_SHPR1
		SHPR2 SCS_SHPR2
		SHPR3 SCS_SHPR3
		SHCSR SCS_SHCSR
		CFSR  SCS_CFSR
		HFSR  SCS_HFSR
		DFSR  SCS_DFSR
		MMFAR SCS_MMFAR
		BFAR  SCS_BFAR
		AFSR  SCS_AFSR
		_     [18]uint32
		CPACR SCS_CPACR
	}

	SCS_CPUID uint32
	SCS_ICSR  uint32
	SCS_VTOR  uint32
	SCS_AIRCR uint32
	SCS_SCR   uint32
	SCS_CCR   uint32
	SCS_SHPR1 uint32
	SCS_SHPR2 uint32
	SCS_SHPR3 uint32
	SCS_SHCSR uint32
	SCS_CFSR  uint32
	SCS_HFSR  uint32
	SCS_DFSR  uint32
	SCS_MMFAR uint32
	SCS_BFAR  uint32
	SCS_AFSR  uint32
	SCS_CPACR uint32
)

func (reg *SCS_ICSR) GetPENDSVSET() bool {
	v := volatile.LoadUint32((*uint32)(reg))
	return v&(0x1<<28) != 0
}

func (reg *SCS_ICSR) SetPENDSVSET(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 28
	} else {
		value &= ^(value & (0x1 << 28))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SCS_ICSR) SetPENDSVCLR(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 27
	} else {
		value &= ^(value & (0x1 << 27))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SCS_ICSR) GetPENDSTSET() bool {
	v := volatile.LoadUint32((*uint32)(reg))
	return v&(0x1<<26) != 0
}

func (reg *SCS_ICSR) SetPENDSTSET(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 26
	} else {
		value &= ^(value & (0x1 << 26))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (reg *SCS_ICSR) SetPENDSTCLR(enable bool) {
	value := volatile.LoadUint32((*uint32)(reg))
	if enable {
		value |= 0x1 << 25
	} else {
		value &= ^(value & (0x1 << 25))
	}
	volatile.StoreUint32((*uint32)(reg), value)
}

func (s *SCS_SHPR1) GetPRI_4() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 0) & 0xFF)
}

func (s *SCS_SHPR1) SetPRI_4(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 0))
	v |= uint32(value) << 0
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR1) GetPRI_5() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 8) & 0xFF)
}

func (s *SCS_SHPR1) SetPRI_5(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 8))
	v |= uint32(value) << 8
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR1) GetPRI_6() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 16) & 0xFF)
}

func (s *SCS_SHPR1) SetPRI_6(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 16))
	v |= uint32(value) << 16
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR2) GetPRI_11() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 24) & 0xFF)
}

func (s *SCS_SHPR2) SetPRI_11(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 24))
	v |= uint32(value) << 24
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR3) GetPRI_12() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 0) & 0xFF)
}

func (s *SCS_SHPR3) SetPRI_12(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 0))
	v |= uint32(value) << 0
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR3) GetPRI_14() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 16) & 0xFF)
}

func (s *SCS_SHPR3) SetPRI_14(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 16))
	v |= uint32(value) << 16
	volatile.StoreUint32((*uint32)(s), v)
}

func (s *SCS_SHPR3) GetPRI_15() uint8 {
	v := volatile.LoadUint32((*uint32)(s))
	return uint8((v >> 24) & 0xFF)
}

func (s *SCS_SHPR3) SetPRI_15(value uint8) {
	v := volatile.LoadUint32((*uint32)(s))
	v &= ^(v & (0xFF << 24))
	v |= uint32(value) << 24
	volatile.StoreUint32((*uint32)(s), v)
}
