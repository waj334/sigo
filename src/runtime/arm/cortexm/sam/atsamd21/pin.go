package atsamd21

import (
	"peripheral"
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
	"unsafe"
	"volatile"
)

type Pin uint32

const (
	SERCOM0 Pin = 0x0000_0000
	SERCOM1 Pin = 0x1000_0000
	SERCOM2 Pin = 0x2000_0000
	SERCOM3 Pin = 0x3000_0000
	SERCOM4 Pin = 0x4000_0000
	SERCOM5 Pin = 0x5000_0000

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
	PA30 Pin = 0x0000_001E | NoPAD | AltPAD2 | AltSERCOM3
	PA31 Pin = 0x0000_001F | NoPAD | AltPAD3 | AltSERCOM3
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
	PB22 Pin = 0x0000_0116 | NoPAD | AltPAD2 | AltSERCOM5
	PB23 Pin = 0x0000_0117 | NoPAD | AltPAD3 | AltSERCOM5
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
)

const (
	Input  peripheral.PinDirection = 0
	Output peripheral.PinDirection = 1
)

const (
	NoEdge peripheral.PinIRQMode = iota
	RisingEdge
	FallingEdge
	BothEdges
	HighLevel
	LowLevel
)

const (
	NoPull peripheral.PinPullMode = iota
	PullUp
	PullDown
)

var (
	handlerFuncs [16]func(Pin)
	handlerPins  [16]Pin
)

func (p Pin) High() {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	portgroup.OUT.SetOUT(1 << (p & 0xFF))
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

func (p Pin) SetInterrupt(mode peripheral.PinIRQMode, handler func(Pin)) {
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
	chip.EIC.CTRL.SetENABLE(true)
	for chip.EIC.STATUS.GetSYNCBUSY() {
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

func (p Pin) SetDirection(dir peripheral.PinDirection) {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if dir == Input {
		portgroup.DIRCLR.SetDIRCLR(1 << (p & 0xFF))
		portgroup.CTRL.SetSAMPLING(1 << (p & 0xFF))
	} else if dir == Output {
		portgroup.DIRSET.SetDIRSET(1 << (p & 0xFF))
	}
	portgroup.PINCFG[p&0xFF].SetINEN(true)
}

func (p Pin) GetDirection() peripheral.PinDirection {
	portgroup := &chip.PORT.GROUP[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.DIR.GetDIR() == 0 {
		return Output
	}
	return Input
}

func (p Pin) SetPullMode(mode peripheral.PinPullMode) {
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

func (p Pin) GetPullMode() peripheral.PinPullMode {
	return 0
}

func (p Pin) GetSERCOM() SERCOM {
	s := int(p>>28) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return SERCOM(s)
}

func (p Pin) GetAltSERCOM() SERCOM {
	s := int(p>>20) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return SERCOM(s)
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
	volatile.StoreUint16((*uint16)(unsafe.Pointer(&chip.EIC.INTENSET)), 1<<eic)
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
