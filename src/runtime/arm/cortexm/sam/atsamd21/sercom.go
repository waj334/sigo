//go:build atsamd21

package atsamd21

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamd21/support/pm"
	"runtime/arm/cortexm/sam/atsamd21/support/sercom/i2cm"
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
		pm.Pm.Apbcmask.SetSercom0(enable)
	case 1:
		pm.Pm.Apbcmask.SetSercom1(enable)
	case 2:
		pm.Pm.Apbcmask.SetSercom2(enable)
	case 3:
		pm.Pm.Apbcmask.SetSercom3(enable)
	case 4:
		pm.Pm.Apbcmask.SetSercom4(enable)
	case 5:
		pm.Pm.Apbcmask.SetSercom5(enable)
	}
}

func (s SERCOM) Baud(hz uint) uint8 {
	return uint8((SercomRefFrequency / (2 * uint32(hz))) - 1)
}

func (s SERCOM) BaudFP(hz uint) (uint16, uint8) {
	ratio := (uint64(SercomRefFrequency) * uint64(1000)) / (uint64(hz) * 16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000
	return uint16(baud), uint8(fp)
}

func (s SERCOM) Synchronize() {
	for i2cm.I2cm[s].Syncbusy.GetEnable() {
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

//sigo:interrupt sercom0Handler Sercom0Handler
func sercom0Handler() {
	if SERCOM0HandlerFunc != nil {
		SERCOM0HandlerFunc()
	}
}

//sigo:interrupt sercom1Handler Sercom1Handler
func sercom1Handler() {
	if SERCOM1HandlerFunc != nil {
		SERCOM1HandlerFunc()
	}
}

//sigo:interrupt sercom2Handler Sercom2Handler
func sercom2Handler() {
	if SERCOM2HandlerFunc != nil {
		SERCOM2HandlerFunc()
	}
}

//sigo:interrupt sercom3Handler Sercom3Handler
func sercom3Handler() {
	if SERCOM3HandlerFunc != nil {
		SERCOM3HandlerFunc()
	}
}

//sigo:interrupt sercom4Handler Sercom4Handler
func sercom4Handler() {
	if SERCOM4HandlerFunc != nil {
		SERCOM4HandlerFunc()
	}
}

//sigo:interrupt sercom5Handler Sercom5Handler
func sercom5Handler() {
	if SERCOM5HandlerFunc != nil {
		SERCOM5HandlerFunc()
	}
}
