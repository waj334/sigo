package atsamd21

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
)

type SERCOM int
type SERCOMHandler func()

var (
	SERCOM0HandlerFunc SERCOMHandler
	SERCOM1HandlerFunc SERCOMHandler
	SERCOM2HandlerFunc SERCOMHandler
	SERCOM3HandlerFunc SERCOMHandler
	SERCOM4HandlerFunc SERCOMHandler
	SERCOM5HandlerFunc SERCOMHandler

	SERCOMHandlers = [6]*SERCOMHandler{
		&SERCOM0HandlerFunc,
		&SERCOM1HandlerFunc,
		&SERCOM2HandlerFunc,
		&SERCOM3HandlerFunc,
		&SERCOM4HandlerFunc,
		&SERCOM5HandlerFunc,
	}
)

func (s SERCOM) SetPMEnabled(enable bool) {
	// Enabled the SERCOM in PM
	switch s {
	case 0:
		chip.PM.APBCMASK.SetSERCOM0(enable)
	case 1:
		chip.PM.APBCMASK.SetSERCOM1(enable)
	case 2:
		chip.PM.APBCMASK.SetSERCOM2(enable)
	case 3:
		chip.PM.APBCMASK.SetSERCOM3(enable)
	case 4:
		chip.PM.APBCMASK.SetSERCOM4(enable)
	case 5:
		chip.PM.APBCMASK.SetSERCOM5(enable)
	}
}

func (s SERCOM) Baud(hz uint) uint8 {
	return uint8((SERCOM_REF_FREQUENCY / (2 * uint32(hz))) - 1)
}

func (s SERCOM) BaudFP(hz uint) (uint16, uint8) {
	ratio := (uint64(SERCOM_REF_FREQUENCY) * uint64(1000)) / (uint64(hz) * 16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000
	return uint16(baud), uint8(fp)
}

func (s SERCOM) Synchronize() {
	for chip.SERCOM_I2CM[s].SYNCBUSY.GetENABLE() {
		// Wait for SERCOM sync
	}
}

func (s SERCOM) Irq() cortexm.Interrupt {
	return IRQ_SERCOM0 + cortexm.Interrupt(s)
}

func (s SERCOM) SetHandler(fn func()) {
	SERCOMHandlers[s].Set(fn)
}

func (s *SERCOMHandler) Set(fn func()) {
	*s = fn
}

//sigo:interrupt SERCOM0_Handler SERCOM0_Handler
func SERCOM0_Handler() {
	if SERCOM0HandlerFunc != nil {
		SERCOM0HandlerFunc()
	}
}

//sigo:interrupt SERCOM1_Handler SERCOM1_Handler
func SERCOM1_Handler() {
	if SERCOM1HandlerFunc != nil {
		SERCOM1HandlerFunc()
	}
}

//sigo:interrupt SERCOM2_Handler SERCOM2_Handler
func SERCOM2_Handler() {
	if SERCOM2HandlerFunc != nil {
		SERCOM2HandlerFunc()
	}
}

//sigo:interrupt SERCOM3_Handler SERCOM3_Handler
func SERCOM3_Handler() {
	if SERCOM3HandlerFunc != nil {
		SERCOM3HandlerFunc()
	}
}

//sigo:interrupt SERCOM4_Handler SERCOM4_Handler
func SERCOM4_Handler() {
	if SERCOM4HandlerFunc != nil {
		SERCOM4HandlerFunc()
	}
}

//sigo:interrupt SERCOM5_Handler SERCOM5_Handler
func SERCOM5_Handler() {
	if SERCOM5HandlerFunc != nil {
		SERCOM5HandlerFunc()
	}
}
