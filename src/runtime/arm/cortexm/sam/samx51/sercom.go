package samx51

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
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
		chip.MCLK.APBAMASK.SetSERCOM0(enable)
	case 1:
		chip.MCLK.APBAMASK.SetSERCOM1(enable)
	case 2:
		chip.MCLK.APBBMASK.SetSERCOM2(enable)
	case 3:
		chip.MCLK.APBBMASK.SetSERCOM3(enable)
	case 4:
		chip.MCLK.APBDMASK.SetSERCOM4(enable)
	case 5:
		chip.MCLK.APBDMASK.SetSERCOM5(enable)
	case 6:
		chip.MCLK.APBDMASK.SetSERCOM6(enable)
	case 7:
		chip.MCLK.APBDMASK.SetSERCOM7(enable)
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

//sigo:interrupt SERCOM0_0_Handler SERCOM0_0_Handler
func SERCOM0_0_Handler() {
	if SERCOM00HandlerFunc != nil {
		SERCOM00HandlerFunc()
	}
}

//sigo:interrupt SERCOM0_1_Handler SERCOM0_1_Handler
func SERCOM0_1_Handler() {
	if SERCOM01HandlerFunc != nil {
		SERCOM01HandlerFunc()
	}
}

//sigo:interrupt SERCOM0_2_Handler SERCOM0_2_Handler
func SERCOM0_2_Handler() {
	if SERCOM02HandlerFunc != nil {
		SERCOM02HandlerFunc()
	}
}

//sigo:interrupt SERCOM0_3_Handler SERCOM0_3_Handler
func SERCOM0_3_Handler() {
	if SERCOM03HandlerFunc != nil {
		SERCOM03HandlerFunc()
	}
}

//sigo:interrupt SERCOM1_0_Handler SERCOM1_0_Handler
func SERCOM1_0_Handler() {
	if SERCOM10HandlerFunc != nil {
		SERCOM10HandlerFunc()
	}
}

//sigo:interrupt SERCOM1_1_Handler SERCOM1_1_Handler
func SERCOM1_1_Handler() {
	if SERCOM11HandlerFunc != nil {
		SERCOM11HandlerFunc()
	}
}

//sigo:interrupt SERCOM1_2_Handler SERCOM1_2_Handler
func SERCOM1_2_Handler() {
	if SERCOM12HandlerFunc != nil {
		SERCOM12HandlerFunc()
	}
}

//sigo:interrupt SERCOM1_3_Handler SERCOM1_3_Handler
func SERCOM1_3_Handler() {
	if SERCOM13HandlerFunc != nil {
		SERCOM13HandlerFunc()
	}
}

//sigo:interrupt SERCOM2_0_Handler SERCOM2_0_Handler
func SERCOM2_0_Handler() {
	if SERCOM20HandlerFunc != nil {
		SERCOM20HandlerFunc()
	}
}

//sigo:interrupt SERCOM2_1_Handler SERCOM2_1_Handler
func SERCOM2_1_Handler() {
	if SERCOM21HandlerFunc != nil {
		SERCOM21HandlerFunc()
	}
}

//sigo:interrupt SERCOM2_2_Handler SERCOM2_2_Handler
func SERCOM2_2_Handler() {
	if SERCOM22HandlerFunc != nil {
		SERCOM22HandlerFunc()
	}
}

//sigo:interrupt SERCOM2_3_Handler SERCOM2_3_Handler
func SERCOM2_3_Handler() {
	if SERCOM23HandlerFunc != nil {
		SERCOM23HandlerFunc()
	}
}

//sigo:interrupt SERCOM3_0_Handler SERCOM3_0_Handler
func SERCOM3_0_Handler() {
	if SERCOM30HandlerFunc != nil {
		SERCOM30HandlerFunc()
	}
}

//sigo:interrupt SERCOM3_1_Handler SERCOM3_1_Handler
func SERCOM3_1_Handler() {
	if SERCOM31HandlerFunc != nil {
		SERCOM31HandlerFunc()
	}
}

//sigo:interrupt SERCOM3_2_Handler SERCOM3_2_Handler
func SERCOM3_2_Handler() {
	if SERCOM32HandlerFunc != nil {
		SERCOM32HandlerFunc()
	}
}

//sigo:interrupt SERCOM3_3_Handler SERCOM3_3_Handler
func SERCOM3_3_Handler() {
	if SERCOM33HandlerFunc != nil {
		SERCOM33HandlerFunc()
	}
}

//sigo:interrupt SERCOM4_0_Handler SERCOM4_0_Handler
func SERCOM4_0_Handler() {
	if SERCOM40HandlerFunc != nil {
		SERCOM40HandlerFunc()
	}
}

//sigo:interrupt SERCOM4_1_Handler SERCOM4_1_Handler
func SERCOM4_1_Handler() {
	if SERCOM41HandlerFunc != nil {
		SERCOM41HandlerFunc()
	}
}

//sigo:interrupt SERCOM4_2_Handler SERCOM4_2_Handler
func SERCOM4_2_Handler() {
	if SERCOM42HandlerFunc != nil {
		SERCOM42HandlerFunc()
	}
}

//sigo:interrupt SERCOM4_3_Handler SERCOM4_3_Handler
func SERCOM4_3_Handler() {
	if SERCOM43HandlerFunc != nil {
		SERCOM43HandlerFunc()
	}
}

//sigo:interrupt SERCOM5_0_Handler SERCOM5_0_Handler
func SERCOM5_0_Handler() {
	if SERCOM50HandlerFunc != nil {
		SERCOM50HandlerFunc()
	}
}

//sigo:interrupt SERCOM5_1_Handler SERCOM5_1_Handler
func SERCOM5_1_Handler() {
	if SERCOM51HandlerFunc != nil {
		SERCOM51HandlerFunc()
	}
}

//sigo:interrupt SERCOM5_2_Handler SERCOM5_2_Handler
func SERCOM5_2_Handler() {
	if SERCOM52HandlerFunc != nil {
		SERCOM52HandlerFunc()
	}
}

//sigo:interrupt SERCOM5_3_Handler SERCOM5_3_Handler
func SERCOM5_3_Handler() {
	if SERCOM53HandlerFunc != nil {
		SERCOM53HandlerFunc()
	}
}

//sigo:interrupt SERCOM6_0_Handler SERCOM6_0_Handler
func SERCOM6_0_Handler() {
	if SERCOM60HandlerFunc != nil {
		SERCOM60HandlerFunc()
	}
}

//sigo:interrupt SERCOM6_1_Handler SERCOM6_1_Handler
func SERCOM6_1_Handler() {
	if SERCOM61HandlerFunc != nil {
		SERCOM61HandlerFunc()
	}
}

//sigo:interrupt SERCOM6_2_Handler SERCOM6_2_Handler
func SERCOM6_2_Handler() {
	if SERCOM62HandlerFunc != nil {
		SERCOM62HandlerFunc()
	}
}

//sigo:interrupt SERCOM6_3_Handler SERCOM6_3_Handler
func SERCOM6_3_Handler() {
	if SERCOM63HandlerFunc != nil {
		SERCOM63HandlerFunc()
	}
}

//sigo:interrupt SERCOM7_0_Handler SERCOM7_0_Handler
func SERCOM7_0_Handler() {
	if SERCOM70HandlerFunc != nil {
		SERCOM70HandlerFunc()
	}
}

//sigo:interrupt SERCOM7_1_Handler SERCOM7_1_Handler
func SERCOM7_1_Handler() {
	if SERCOM71HandlerFunc != nil {
		SERCOM71HandlerFunc()
	}
}

//sigo:interrupt SERCOM7_2_Handler SERCOM7_2_Handler
func SERCOM7_2_Handler() {
	if SERCOM72HandlerFunc != nil {
		SERCOM72HandlerFunc()
	}
}

//sigo:interrupt SERCOM7_3_Handler SERCOM7_3_Handler
func SERCOM7_3_Handler() {
	if SERCOM73HandlerFunc != nil {
		SERCOM73HandlerFunc()
	}
}
