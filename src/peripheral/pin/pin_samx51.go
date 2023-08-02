//go:build samx51 && !generic

package pin

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samx51"
)

type Pin uint32

const (
	SERCOM0 Pin = 0x0000_0000
	SERCOM1 Pin = 0x1000_0000
	SERCOM2 Pin = 0x2000_0000
	SERCOM3 Pin = 0x3000_0000
	SERCOM4 Pin = 0x4000_0000
	SERCOM5 Pin = 0x5000_0000
	SERCOM6 Pin = 0x6000_0000
	SERCOM7 Pin = 0x7000_0000

	PAD0 Pin = 0x0000_0000
	PAD1 Pin = 0x0100_0000
	PAD2 Pin = 0x0200_0000
	PAD3 Pin = 0x0300_0000

	AltSERCOM0 Pin = 0x0000_0000
	AltSERCOM1 Pin = 0x0010_0000
	AltSERCOM2 Pin = 0x0020_0000
	AltSERCOM3 Pin = 0x0030_0000
	AltSERCOM4 Pin = 0x0040_0000
	AltSERCOM5 Pin = 0x0050_0000
	AltSERCOM6 Pin = 0x0060_0000
	AltSERCOM7 Pin = 0x0070_0000

	AltPAD0 Pin = 0x0000_0000
	AltPAD1 Pin = 0x0001_0000
	AltPAD2 Pin = 0x0002_0000
	AltPAD3 Pin = 0x0003_0000

	NoPAD Pin = 0xFF00_0000
	NoALT Pin = 0x00FF_0000
	NoPin Pin = 0
)

// Pin group 0
const (
	PA00 Pin = 0x0000_0000 | AltPAD0 | AltSERCOM1
	PA01 Pin = 0x0000_0001 | AltPAD1 | AltSERCOM1
	PA02 Pin = 0x0000_0002 | NoPAD | NoALT
	PA03 Pin = 0x0000_0003 | NoPAD | NoALT
	PA04 Pin = 0x0000_0004 | AltPAD0 | AltSERCOM0 | NoPAD
	PA05 Pin = 0x0000_0005 | AltPAD1 | AltSERCOM0 | NoPAD
	PA06 Pin = 0x0000_0006 | AltPAD2 | AltSERCOM0 | NoPAD
	PA07 Pin = 0x0000_0007 | AltPAD3 | AltSERCOM0 | NoPAD
	PA08 Pin = 0x0000_0008 | PAD0 | SERCOM0 | AltPAD1 | AltSERCOM2
	PA09 Pin = 0x0000_0009 | PAD1 | SERCOM0 | AltPAD0 | AltSERCOM2
	PA10 Pin = 0x0000_000A | PAD2 | SERCOM0 | AltPAD2 | AltSERCOM2
	PA11 Pin = 0x0000_000B | PAD3 | SERCOM0 | AltPAD3 | AltSERCOM2
	PA12 Pin = 0x0000_000C | PAD0 | SERCOM2 | AltPAD1 | AltSERCOM4
	PA13 Pin = 0x0000_000D | PAD1 | SERCOM2 | AltPAD0 | AltSERCOM4
	PA14 Pin = 0x0000_000E | PAD2 | SERCOM2 | AltPAD2 | AltSERCOM4
	PA15 Pin = 0x0000_000F | PAD3 | SERCOM2 | AltPAD3 | AltSERCOM4
	PA16 Pin = 0x0000_0010 | PAD0 | SERCOM1 | AltPAD1 | AltSERCOM3
	PA17 Pin = 0x0000_0011 | PAD1 | SERCOM1 | AltPAD0 | AltSERCOM3
	PA18 Pin = 0x0000_0012 | PAD2 | SERCOM1 | AltPAD2 | AltSERCOM3
	PA19 Pin = 0x0000_0013 | PAD3 | SERCOM1 | AltPAD3 | AltSERCOM3
	PA20 Pin = 0x0000_0014 | PAD2 | SERCOM5 | AltPAD2 | AltSERCOM3
	PA21 Pin = 0x0000_0015 | PAD3 | SERCOM5 | AltPAD3 | AltSERCOM3
	PA22 Pin = 0x0000_0016 | PAD0 | SERCOM3 | AltPAD1 | AltSERCOM5
	PA23 Pin = 0x0000_0017 | PAD1 | SERCOM3 | AltPAD0 | AltSERCOM5
	PA24 Pin = 0x0000_0018 | PAD2 | SERCOM3 | AltPAD2 | AltSERCOM5
	PA25 Pin = 0x0000_0019 | PAD3 | SERCOM3 | AltPAD3 | AltSERCOM5
	PA26 Pin = 0x0000_001A | NoPAD | NoALT
	PA27 Pin = 0x0000_001B | NoPAD | NoALT
	PA28 Pin = 0x0000_001C | NoPAD | NoALT
	PA29 Pin = 0x0000_001D | NoPAD | NoALT
	PA30 Pin = 0x0000_001E | PAD2 | SERCOM7 | AltPAD2 | AltSERCOM3
	PA31 Pin = 0x0000_001F | PAD3 | SERCOM7 | AltPAD3 | AltSERCOM3
)

// Pin group 1
const (
	PB00 Pin = 0x0000_0100 | AltPAD2 | AltSERCOM5 | NoPAD
	PB01 Pin = 0x0000_0011 | AltPAD3 | AltSERCOM5 | NoPAD
	PB02 Pin = 0x0000_0102 | AltPAD0 | AltSERCOM5 | NoPAD
	PB03 Pin = 0x0000_0103 | AltPAD1 | AltSERCOM5 | NoPAD
	PB04 Pin = 0x0000_0104 | NoPAD | NoALT
	PB05 Pin = 0x0000_0105 | NoPAD | NoALT
	PB06 Pin = 0x0000_0106 | NoPAD | NoALT
	PB07 Pin = 0x0000_0107 | NoPAD | NoALT
	PB08 Pin = 0x0000_0108 | AltPAD0 | AltSERCOM4 | NoPAD
	PB09 Pin = 0x0000_0109 | AltPAD1 | AltSERCOM4 | NoPAD
	PB10 Pin = 0x0000_010A | AltPAD2 | AltSERCOM4 | NoPAD
	PB11 Pin = 0x0000_010B | AltPAD3 | AltSERCOM4 | NoPAD
	PB12 Pin = 0x0000_010C | PAD0 | SERCOM4 | NoALT
	PB13 Pin = 0x0000_010D | PAD1 | SERCOM4 | NoALT
	PB14 Pin = 0x0000_010E | PAD2 | SERCOM4 | NoALT
	PB15 Pin = 0x0000_010F | PAD3 | SERCOM4 | NoALT
	PB16 Pin = 0x0000_0110 | PAD0 | SERCOM5 | NoALT
	PB17 Pin = 0x0000_0111 | PAD1 | SERCOM5 | NoALT
	PB18 Pin = 0x0000_0112 | PAD2 | SERCOM5 | AltPAD2 | AltSERCOM7
	PB19 Pin = 0x0000_0113 | PAD3 | SERCOM5 | AltPAD3 | AltSERCOM7
	PB20 Pin = 0x0000_0114 | PAD0 | SERCOM3 | AltPAD1 | AltSERCOM7
	PB21 Pin = 0x0000_0115 | PAD1 | SERCOM3 | AltPAD0 | AltSERCOM7
	PB22 Pin = 0x0000_0116 | PAD2 | SERCOM1 | AltPAD2 | AltSERCOM5
	PB23 Pin = 0x0000_0117 | PAD3 | SERCOM1 | AltPAD3 | AltSERCOM5
	PB24 Pin = 0x0000_0118 | PAD0 | SERCOM0 | AltPAD1 | AltSERCOM2
	PB25 Pin = 0x0000_0119 | PAD1 | SERCOM0 | AltPAD0 | AltSERCOM2
	PB26 Pin = 0x0000_011A | PAD2 | SERCOM0 | AltPAD2 | AltSERCOM2
	PB27 Pin = 0x0000_011B | PAD3 | SERCOM0 | AltPAD3 | AltSERCOM2
	PB28 Pin = 0x0000_011C | PAD2 | SERCOM2 | AltPAD2 | AltSERCOM4
	PB29 Pin = 0x0000_011D | PAD3 | SERCOM2 | AltPAD3 | AltSERCOM4
	PB30 Pin = 0x0000_011E | PAD0 | SERCOM7 | AltPAD1 | AltSERCOM5
	PB31 Pin = 0x0000_011F | PAD1 | SERCOM7 | AltPAD0 | AltSERCOM5
)

// Pin group 2
const (
	PC00 Pin = 0x0000_0200 | NoPAD | NoALT
	PC01 Pin = 0x0000_0021 | NoPAD | NoALT
	PC02 Pin = 0x0000_0202 | NoPAD | NoALT
	PC03 Pin = 0x0000_0203 | NoPAD | NoALT
	PC04 Pin = 0x0000_0204 | PAD0 | SERCOM6 | NoALT
	PC05 Pin = 0x0000_0205 | PAD1 | SERCOM6 | NoALT
	PC06 Pin = 0x0000_0206 | PAD2 | SERCOM6 | NoALT
	PC07 Pin = 0x0000_0207 | PAD3 | SERCOM6 | NoALT
	PC08 Pin = 0x0000_0208 | NoPAD | NoALT
	PC09 Pin = 0x0000_0209 | NoPAD | NoALT
	PC10 Pin = 0x0000_020A | PAD2 | SERCOM6 | AltPAD2 | AltSERCOM7
	PC11 Pin = 0x0000_020B | PAD3 | SERCOM6 | AltPAD3 | AltSERCOM7
	PC12 Pin = 0x0000_020C | PAD0 | SERCOM7 | AltPAD1 | AltSERCOM6
	PC13 Pin = 0x0000_020D | PAD1 | SERCOM7 | AltPAD0 | AltSERCOM6
	PC14 Pin = 0x0000_020E | PAD2 | SERCOM7 | AltPAD2 | AltSERCOM6
	PC15 Pin = 0x0000_020F | PAD3 | SERCOM7 | AltPAD3 | AltSERCOM6
	PC16 Pin = 0x0000_0210 | PAD0 | SERCOM6 | AltPAD1 | AltSERCOM0
	PC17 Pin = 0x0000_0211 | PAD1 | SERCOM6 | AltPAD0 | AltSERCOM0
	PC18 Pin = 0x0000_0212 | PAD2 | SERCOM6 | AltPAD2 | AltSERCOM0
	PC19 Pin = 0x0000_0213 | PAD3 | SERCOM6 | AltPAD3 | AltSERCOM0
	PC20 Pin = 0x0000_0214 | NoPAD | NoALT
	PC21 Pin = 0x0000_0215 | NoPAD | NoALT
	PC22 Pin = 0x0000_0216 | PAD0 | SERCOM1 | AltPAD1 | AltSERCOM3
	PC23 Pin = 0x0000_0217 | PAD1 | SERCOM1 | AltPAD0 | AltSERCOM3
	PC24 Pin = 0x0000_0218 | PAD2 | SERCOM0 | AltPAD2 | AltSERCOM2
	PC25 Pin = 0x0000_0219 | PAD3 | SERCOM0 | AltPAD3 | AltSERCOM2
	PC26 Pin = 0x0000_021A | NoPAD | NoALT
	PC27 Pin = 0x0000_021B | PAD0 | SERCOM1 | NoALT
	PC28 Pin = 0x0000_021C | PAD1 | SERCOM1 | NoALT
	PC29 Pin = 0x0000_021D | NoPAD | NoALT
	PC30 Pin = 0x0000_021E | NoPAD | NoALT
	PC31 Pin = 0x0000_021F | NoPAD | NoALT
)

// Pin group 3
const (
	PD00 Pin = 0x0000_0300 | NoPAD | NoALT
	PD01 Pin = 0x0000_0031 | NoPAD | NoALT
	PD02 Pin = 0x0000_0302 | NoPAD | NoALT
	PD03 Pin = 0x0000_0303 | NoPAD | NoALT
	PD04 Pin = 0x0000_0304 | NoPAD | NoALT
	PD05 Pin = 0x0000_0305 | NoPAD | NoALT
	PD06 Pin = 0x0000_0306 | NoPAD | NoALT
	PD07 Pin = 0x0000_0307 | NoPAD | NoALT
	PD08 Pin = 0x0000_0308 | PAD0 | SERCOM7 | AltPAD1 | AltSERCOM6
	PD09 Pin = 0x0000_0309 | PAD1 | SERCOM7 | AltPAD0 | AltSERCOM6
	PD10 Pin = 0x0000_030A | PAD2 | SERCOM7 | AltPAD2 | AltSERCOM6
	PD11 Pin = 0x0000_030B | PAD3 | SERCOM7 | AltPAD3 | AltSERCOM6
	PD12 Pin = 0x0000_030C | NoPAD | NoALT
	PD13 Pin = 0x0000_030D | NoPAD | NoALT
	PD14 Pin = 0x0000_030E | NoPAD | NoALT
	PD15 Pin = 0x0000_030F | NoPAD | NoALT
	PD16 Pin = 0x0000_0310 | NoPAD | NoALT
	PD17 Pin = 0x0000_0311 | NoPAD | NoALT
	PD18 Pin = 0x0000_0312 | NoPAD | NoALT
	PD19 Pin = 0x0000_0313 | NoPAD | NoALT
	PD20 Pin = 0x0000_0314 | PAD2 | SERCOM1 | AltPAD2 | AltSERCOM3
	PD21 Pin = 0x0000_0315 | PAD3 | SERCOM1 | AltPAD3 | AltSERCOM3
	PD22 Pin = 0x0000_0316 | NoPAD | NoALT
	PD23 Pin = 0x0000_0317 | NoPAD | NoALT
	PD24 Pin = 0x0000_0318 | NoPAD | NoALT
	PD25 Pin = 0x0000_0319 | NoPAD | NoALT
	PD26 Pin = 0x0000_031A | NoPAD | NoALT
	PD27 Pin = 0x0000_031B | NoPAD | NoALT
	PD28 Pin = 0x0000_031C | NoPAD | NoALT
	PD29 Pin = 0x0000_031D | NoPAD | NoALT
	PD30 Pin = 0x0000_031E | NoPAD | NoALT
	PD31 Pin = 0x0000_031F | NoPAD | NoALT
)

type PMUXFunction uint8

const (
	PMUXFunctionA PMUXFunction = iota
	PMUXFunctionB
	PMUXFunctionC
	PMUXFunctionD
	PMUXFunctionE
	PMUXFunctionF
	PMUXFunctionG
	PMUXFunctionH
	PMUXFunctionI
	PMUXFunctionJ
	PMUXFunctionK
	PMUXFunctionL
	PMUXFunctionM
	PMUXFunctionN
)

const (
	Input  Direction = 0
	Output Direction = 1
)

const (
	NoEdge IRQMode = iota
	RisingEdge
	FallingEdge
	BothEdges
	HighLevel
	LowLevel
)

const (
	NoPull PullMode = iota
	PullUp
	PullDown
)

var (
	handlerFuncs [16]func(Pin)
	handlerPins  [16]Pin
)

func (p Pin) High() {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	portgroup.OUTSET.SetOUTSET(1 << (p & 0xFF))
}

func (p Pin) Low() {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	portgroup.OUTCLR.SetOUTCLR(1 << (p & 0xFF))
}

func (p Pin) Toggle() {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	portgroup.OUTTGL.SetOUTTGL(1 << (p & 0xFF))
}

func (p Pin) Set(on bool) {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if on {
		portgroup.OUTSET.SetOUTSET(1 << (p & 0xFF))
	} else {
		portgroup.OUTCLR.SetOUTCLR(1 << (p & 0xFF))
	}
}

func (p Pin) Get() bool {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.DIR.GetDIR() == 0 {
		// Is input. Return the input value.
		return portgroup.IN.GetIN()&(1<<(p&0xFF)) != 0
	} else {
		// Is output. Return the asserted state.
		return portgroup.OUT.GetOUT()&(1<<(p&0xFF)) != 0
	}
}

func (p Pin) SetInterrupt(mode IRQMode, handler func(Pin)) {
	// Bounds check the mode
	if mode < 0 || mode > 5 {
		panic("invalid mode value")
	}

	// Some pins don't support interrupts
	if p == PA08 {
		return
	}

	// Set up PMUX
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	pmux := int(p&0xFF) / 2
	if (p&0xFF)%2 == 0 {
		// Pin is odd numbered
		portgroup.PMUX[pmux].SetPMUXE(0)
	} else {
		portgroup.PMUX[pmux].SetPMUXO(0)
	}
	portgroup.PINCFG[p&0xFF].SetPMUXEN(true)

	// Determine the EXTINT from the pin number
	exint := int((p & 0xFF) % 16)

	handlerFuncs[exint] = handler
	handlerPins[exint] = p

	// Set the configuration
	config := exint / 8
	pos := exint * 4
	mask := (0x07 << (3 * (exint % 2))) << pos
	configVal := (mode << (3 * (exint % 2))) << pos

	chip.EIC.CONFIG[config] = (chip.EIC.CONFIG[config] & (^chip.EIC_CONFIG_REG(mask))) | chip.EIC_CONFIG_REG(configVal)

	// Enable the interrupt
	chip.EIC.INTENSET |= 1 << exint

	// Enable EIC
	chip.EIC.CTRLA.SetENABLE(true)
	for chip.EIC.SYNCBUSY.GetENABLE() {
	}

	// Enable the interrupt in NVIC
	irq := cortexm.Interrupt(12 + exint)
	//irq.SetPriority(0xC0)
	irq.EnableIRQ()
}

func (p Pin) SetPMUX(mode PMUXFunction, enabled bool) {
	// Set up PMUX
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	pmux := int(p&0xFF) / 2
	if (p&0xFF)%2 == 0 {
		// Pin is odd numbered
		portgroup.PMUX[pmux].SetPMUXE(chip.PORT_PMUX_REG_PMUXE(mode))
	} else {
		portgroup.PMUX[pmux].SetPMUXO(chip.PORT_PMUX_REG_PMUXO(mode))
	}
	portgroup.PINCFG[p&0xFF].SetPMUXEN(enabled)
}

func (p Pin) ClearInterrupt() {
	// Determine the EXTINT from the pin number
	exint := (p & 0xFF) % 16
	if chip.EIC.INTENSET&(1<<exint) != 0 {
		// Disable the interrupt
		chip.EIC.INTENCLR &= 1 << exint

		// Disable PMUX
		portgroup := &chip.PORT.GROUP[p>>8]
		portgroup.PINCFG[p&0xFF].SetPMUXEN(false)

		// Disable the interrupt in NVIC
		irq := cortexm.Interrupt(12 + exint)
		irq.DisableIRQ()

		handlerFuncs[exint] = nil
		handlerPins[exint] = 0x00FF
	}
}

func (p Pin) SetDirection(dir Direction) {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if dir == Input {
		portgroup.DIRCLR.SetDIRCLR(1 << (p & 0xFF))
		portgroup.CTRL.SetSAMPLING(1 << (p & 0xFF))
	} else if dir == Output {
		portgroup.DIRSET.SetDIRSET(1 << (p & 0xFF))
	}
	portgroup.PINCFG[p&0xFF].SetINEN(true)
}

func (p Pin) GetDirection() Direction {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.DIR.GetDIR() == 0 {
		return Output
	}
	return Input
}

func (p Pin) SetPullMode(mode PullMode) {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.DIR.GetDIR() == 0 {
		if mode == PullDown {
			p.Set(false)
			portgroup.PINCFG[p&0xFF].SetPULLEN(true)
		} else if mode == PullUp {
			p.Set(true)
			portgroup.PINCFG[p&0xFF].SetPULLEN(true)
		} else { // NoPull
			portgroup.PINCFG[p&0xFF].SetPULLEN(false)
		}
	}
}

func (p Pin) GetPullMode() PullMode {
	return 0
}

func (p Pin) GetSERCOM() samx51.SERCOM {
	s := int(p>>28) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return samx51.SERCOM(s)
}

func (p Pin) GetAltSERCOM() samx51.SERCOM {
	s := int(p>>20) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return samx51.SERCOM(s)
}

func (p Pin) GetPAD() int {
	s := int(p>>24) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return s
}

func (p Pin) GetAltPAD() int {
	s := int(p>>16) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return s
}

func eicHandler(eic int) {
	if fn := handlerFuncs[eic]; fn != nil {
		fn(handlerPins[eic])
	}
	// Clear the interrupt flag
	chip.EIC.INTFLAG.SetEXTINT(1 << eic)
}

//sigo:interrupt _EIC_EXTINT_0_Handler EIC_EXTINT_0_Handler
func _EIC_EXTINT_0_Handler() {
	eicHandler(0)
}

//sigo:interrupt _EIC_EXTINT_1_Handler EIC_EXTINT_1_Handler
func _EIC_EXTINT_1_Handler() {
	eicHandler(1)
}

//sigo:interrupt _EIC_EXTINT_2_Handler EIC_EXTINT_2_Handler
func _EIC_EXTINT_2_Handler() {
	eicHandler(2)
}

//sigo:interrupt _EIC_EXTINT_3_Handler EIC_EXTINT_3_Handler
func _EIC_EXTINT_3_Handler() {
	eicHandler(3)
}

//sigo:interrupt _EIC_EXTINT_4_Handler EIC_EXTINT_4_Handler
func _EIC_EXTINT_4_Handler() {
	eicHandler(4)
}

//sigo:interrupt _EIC_EXTINT_5_Handler EIC_EXTINT_5_Handler
func _EIC_EXTINT_5_Handler() {
	eicHandler(5)
}

//sigo:interrupt _EIC_EXTINT_6_Handler EIC_EXTINT_6_Handler
func _EIC_EXTINT_6_Handler() {
	eicHandler(6)
}

//sigo:interrupt _EIC_EXTINT_7_Handler EIC_EXTINT_7_Handler
func _EIC_EXTINT_7_Handler() {
	eicHandler(7)
}

//sigo:interrupt _EIC_EXTINT_8_Handler EIC_EXTINT_8_Handler
func _EIC_EXTINT_8_Handler() {
	eicHandler(8)
}

//sigo:interrupt _EIC_EXTINT_9_Handler EIC_EXTINT_9_Handler
func _EIC_EXTINT_9_Handler() {
	eicHandler(9)
}

//sigo:interrupt _EIC_EXTINT_10_Handler EIC_EXTINT_10_Handler
func _EIC_EXTINT_10_Handler() {
	eicHandler(10)
}

//sigo:interrupt _EIC_EXTINT_11_Handler EIC_EXTINT_11_Handler
func _EIC_EXTINT_11_Handler() {
	eicHandler(11)
}

//sigo:interrupt _EIC_EXTINT_12_Handler EIC_EXTINT_12_Handler
func _EIC_EXTINT_12_Handler() {
	eicHandler(12)
}

//sigo:interrupt _EIC_EXTINT_13_Handler EIC_EXTINT_13_Handler
func _EIC_EXTINT_13_Handler() {
	eicHandler(13)
}

//sigo:interrupt _EIC_EXTINT_14_Handler EIC_EXTINT_14_Handler
func _EIC_EXTINT_14_Handler() {
	eicHandler(14)
}

//sigo:interrupt _EIC_EXTINT_15_Handler EIC_EXTINT_15_Handler
func _EIC_EXTINT_15_Handler() {
	eicHandler(15)
}
