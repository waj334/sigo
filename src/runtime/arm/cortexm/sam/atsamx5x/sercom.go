//go:build atsamx5x

package atsamx5x

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamx5x/support/mclk"
	"runtime/arm/cortexm/sam/atsamx5x/support/sercom/i2cm"
)

type SERCOM int
type SERCOMHandler func()

var (
	SERCOM00HandlerFunc SERCOMHandler
	SERCOM01HandlerFunc SERCOMHandler
	SERCOM02HandlerFunc SERCOMHandler
	SERCOM03HandlerFunc SERCOMHandler

	SERCOM10HandlerFunc SERCOMHandler
	SERCOM11HandlerFunc SERCOMHandler
	SERCOM12HandlerFunc SERCOMHandler
	SERCOM13HandlerFunc SERCOMHandler

	SERCOM20HandlerFunc SERCOMHandler
	SERCOM21HandlerFunc SERCOMHandler
	SERCOM22HandlerFunc SERCOMHandler
	SERCOM23HandlerFunc SERCOMHandler

	SERCOM30HandlerFunc SERCOMHandler
	SERCOM31HandlerFunc SERCOMHandler
	SERCOM32HandlerFunc SERCOMHandler
	SERCOM33HandlerFunc SERCOMHandler

	SERCOM40HandlerFunc SERCOMHandler
	SERCOM41HandlerFunc SERCOMHandler
	SERCOM42HandlerFunc SERCOMHandler
	SERCOM43HandlerFunc SERCOMHandler

	SERCOM50HandlerFunc SERCOMHandler
	SERCOM51HandlerFunc SERCOMHandler
	SERCOM52HandlerFunc SERCOMHandler
	SERCOM53HandlerFunc SERCOMHandler

	SERCOM60HandlerFunc SERCOMHandler
	SERCOM61HandlerFunc SERCOMHandler
	SERCOM62HandlerFunc SERCOMHandler
	SERCOM63HandlerFunc SERCOMHandler

	SERCOM70HandlerFunc SERCOMHandler
	SERCOM71HandlerFunc SERCOMHandler
	SERCOM72HandlerFunc SERCOMHandler
	SERCOM73HandlerFunc SERCOMHandler

	SERCOMHandlers = [8][4]*SERCOMHandler{
		{
			&SERCOM00HandlerFunc,
			&SERCOM01HandlerFunc,
			&SERCOM02HandlerFunc,
			&SERCOM03HandlerFunc,
		},
		{
			&SERCOM10HandlerFunc,
			&SERCOM11HandlerFunc,
			&SERCOM12HandlerFunc,
			&SERCOM13HandlerFunc,
		},
		{
			&SERCOM20HandlerFunc,
			&SERCOM21HandlerFunc,
			&SERCOM22HandlerFunc,
			&SERCOM23HandlerFunc,
		},
		{
			&SERCOM30HandlerFunc,
			&SERCOM31HandlerFunc,
			&SERCOM32HandlerFunc,
			&SERCOM33HandlerFunc,
		},
		{
			&SERCOM40HandlerFunc,
			&SERCOM41HandlerFunc,
			&SERCOM42HandlerFunc,
			&SERCOM43HandlerFunc,
		},
		{
			&SERCOM50HandlerFunc,
			&SERCOM51HandlerFunc,
			&SERCOM52HandlerFunc,
			&SERCOM53HandlerFunc,
		},
		{
			&SERCOM60HandlerFunc,
			&SERCOM61HandlerFunc,
			&SERCOM62HandlerFunc,
			&SERCOM63HandlerFunc,
		},
		{
			&SERCOM70HandlerFunc,
			&SERCOM71HandlerFunc,
			&SERCOM72HandlerFunc,
			&SERCOM73HandlerFunc,
		},
	}
)

func (s SERCOM) SetEnabled(enable bool) {
	// Enabled the SERCOM in MCLK
	switch s {
	case 0:
		mclk.Mclk.Apbamask.SetSercom0(enable)
	case 1:
		mclk.Mclk.Apbamask.SetSercom1(enable)
	case 2:
		mclk.Mclk.Apbbmask.SetSercom2(enable)
	case 3:
		mclk.Mclk.Apbbmask.SetSercom3(enable)
	case 4:
		mclk.Mclk.Apbdmask.SetSercom4(enable)
	case 5:
		mclk.Mclk.Apbdmask.SetSercom5(enable)
	case 6:
		mclk.Mclk.Apbdmask.SetSercom6(enable)
	case 7:
		mclk.Mclk.Apbdmask.SetSercom7(enable)
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
	// NOTE: The syncbusy flag can be accessed from any of the serial peripherals.
	for i2cm.I2cm[s].Syncbusy.GetEnable() {
		// Wait for SERCOM sync
	}
}

func (s SERCOM) Irq0() cortexm.Interrupt {
	irqBase := 46 + s*4
	return cortexm.Interrupt(irqBase)
}

func (s SERCOM) Irq1() cortexm.Interrupt {
	return s.Irq0() + 1
}

func (s SERCOM) Irq2() cortexm.Interrupt {
	return s.Irq0() + 2
}

func (s SERCOM) IrqMisc() cortexm.Interrupt {
	return s.Irq0() + 3
}

func (s *SERCOMHandler) Set(fn func()) {
	*s = fn
}

//sigo:interrupt Sercom00Handler Sercom00Handler
func Sercom00Handler() {
	if SERCOM00HandlerFunc != nil {
		SERCOM00HandlerFunc()
	}
}

//sigo:interrupt Sercom01Handler Sercom01Handler
func Sercom01Handler() {
	if SERCOM01HandlerFunc != nil {
		SERCOM01HandlerFunc()
	}
}

//sigo:interrupt Sercom02Handler Sercom02Handler
func Sercom02Handler() {
	if SERCOM02HandlerFunc != nil {
		SERCOM02HandlerFunc()
	}
}

//sigo:interrupt Sercom0OtherHandler Sercom0OtherHandler
func Sercom0OtherHandler() {
	if SERCOM03HandlerFunc != nil {
		SERCOM03HandlerFunc()
	}
}

//sigo:interrupt Sercom10Handler Sercom10Handler
func Sercom10Handler() {
	if SERCOM10HandlerFunc != nil {
		SERCOM10HandlerFunc()
	}
}

//sigo:interrupt Sercom11Handler Sercom11Handler
func Sercom11Handler() {
	if SERCOM11HandlerFunc != nil {
		SERCOM11HandlerFunc()
	}
}

//sigo:interrupt Sercom12Handler Sercom12Handler
func Sercom12Handler() {
	if SERCOM12HandlerFunc != nil {
		SERCOM12HandlerFunc()
	}
}

//sigo:interrupt Sercom1OtherHandler Sercom1OtherHandler
func Sercom1OtherHandler() {
	if SERCOM13HandlerFunc != nil {
		SERCOM13HandlerFunc()
	}
}

//sigo:interrupt Sercom20Handler Sercom20Handler
func Sercom20Handler() {
	if SERCOM20HandlerFunc != nil {
		SERCOM20HandlerFunc()
	}
}

//sigo:interrupt Sercom21Handler Sercom21Handler
func Sercom21Handler() {
	if SERCOM21HandlerFunc != nil {
		SERCOM21HandlerFunc()
	}
}

//sigo:interrupt Sercom22Handler Sercom22Handler
func Sercom22Handler() {
	if SERCOM22HandlerFunc != nil {
		SERCOM22HandlerFunc()
	}
}

//sigo:interrupt Sercom2OtherHandler Sercom2OtherHandler
func Sercom2OtherHandler() {
	if SERCOM23HandlerFunc != nil {
		SERCOM23HandlerFunc()
	}
}

//sigo:interrupt Sercom30Handler Sercom30Handler
func Sercom30Handler() {
	if SERCOM30HandlerFunc != nil {
		SERCOM30HandlerFunc()
	}
}

//sigo:interrupt Sercom31Handler Sercom31Handler
func Sercom31Handler() {
	if SERCOM31HandlerFunc != nil {
		SERCOM31HandlerFunc()
	}
}

//sigo:interrupt Sercom32Handler Sercom32Handler
func Sercom32Handler() {
	if SERCOM32HandlerFunc != nil {
		SERCOM32HandlerFunc()
	}
}

//sigo:interrupt Sercom3OtherHandler Sercom3OtherHandler
func Sercom3OtherHandler() {
	if SERCOM33HandlerFunc != nil {
		SERCOM33HandlerFunc()
	}
}

//sigo:interrupt Sercom40Handler Sercom40Handler
func Sercom40Handler() {
	if SERCOM40HandlerFunc != nil {
		SERCOM40HandlerFunc()
	}
}

//sigo:interrupt Sercom41Handler Sercom41Handler
func Sercom41Handler() {
	if SERCOM41HandlerFunc != nil {
		SERCOM41HandlerFunc()
	}
}

//sigo:interrupt Sercom42Handler Sercom42Handler
func Sercom42Handler() {
	if SERCOM42HandlerFunc != nil {
		SERCOM42HandlerFunc()
	}
}

//sigo:interrupt Sercom4OtherHandler Sercom4OtherHandler
func Sercom4OtherHandler() {
	if SERCOM43HandlerFunc != nil {
		SERCOM43HandlerFunc()
	}
}

//sigo:interrupt Sercom50Handler Sercom50Handler
func Sercom50Handler() {
	if SERCOM50HandlerFunc != nil {
		SERCOM50HandlerFunc()
	}
}

//sigo:interrupt Sercom51Handler Sercom51Handler
func Sercom51Handler() {
	if SERCOM51HandlerFunc != nil {
		SERCOM51HandlerFunc()
	}
}

//sigo:interrupt Sercom52Handler Sercom52Handler
func Sercom52Handler() {
	if SERCOM52HandlerFunc != nil {
		SERCOM52HandlerFunc()
	}
}

//sigo:interrupt Sercom5OtherHandler Sercom5OtherHandler
func Sercom5OtherHandler() {
	if SERCOM53HandlerFunc != nil {
		SERCOM53HandlerFunc()
	}
}

//sigo:interrupt Sercom60Handler Sercom60Handler
func Sercom60Handler() {
	if SERCOM60HandlerFunc != nil {
		SERCOM60HandlerFunc()
	}
}

//sigo:interrupt Sercom61Handler Sercom61Handler
func Sercom61Handler() {
	if SERCOM61HandlerFunc != nil {
		SERCOM61HandlerFunc()
	}
}

//sigo:interrupt Sercom62Handler Sercom62Handler
func Sercom62Handler() {
	if SERCOM62HandlerFunc != nil {
		SERCOM62HandlerFunc()
	}
}

//sigo:interrupt Sercom6OtherHandler Sercom6OtherHandler
func Sercom6OtherHandler() {
	if SERCOM63HandlerFunc != nil {
		SERCOM63HandlerFunc()
	}
}

//sigo:interrupt Sercom70Handler Sercom70Handler
func Sercom70Handler() {
	if SERCOM70HandlerFunc != nil {
		SERCOM70HandlerFunc()
	}
}

//sigo:interrupt Sercom71Handler Sercom71Handler
func Sercom71Handler() {
	if SERCOM71HandlerFunc != nil {
		SERCOM71HandlerFunc()
	}
}

//sigo:interrupt Sercom72Handler Sercom72Handler
func Sercom72Handler() {
	if SERCOM72HandlerFunc != nil {
		SERCOM72HandlerFunc()
	}
}

//sigo:interrupt Sercom7OtherHandler Sercom7OtherHandler
func Sercom7OtherHandler() {
	if SERCOM73HandlerFunc != nil {
		SERCOM73HandlerFunc()
	}
}
