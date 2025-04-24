//go:build atsamd21 && !generic

package pin

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamd21"
	"runtime/arm/cortexm/sam/atsamd21/support/eic"
	"runtime/arm/cortexm/sam/atsamd21/support/port"
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
	Input  Mode = 0
	Output Mode = 1
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
	handlerFuncs [16]func()
)

func (p Pin) High() {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	portgroup.Out.SetOut(1 << (p & 0xFF))
}

func (p Pin) Low() {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	portgroup.Outclr.SetOutclr(1 << (p & 0xFF))
}

func (p Pin) Toggle() {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	portgroup.Outtgl.SetOuttgl(1 << (p & 0xFF))
}

func (p Pin) Set(on bool) {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	if on {
		portgroup.Outset.SetOutset(1 << (p & 0xFF))
	} else {
		portgroup.Outclr.SetOutclr(1 << (p & 0xFF))
	}
}

func (p Pin) Get() bool {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.Dir.GetDir() == 0 {
		// Is input. Return the input value.
		return portgroup.In.GetIn()&(1<<(p&0xFF)) != 0
	} else {
		// Is output. Return the asserted state.
		return portgroup.Out.GetOut()&(1<<(p&0xFF)) != 0
	}
}

func (p Pin) SetInterrupt(mode IRQMode, handler func()) {
	// Bounds check the mode
	if mode < 0 || mode > 5 {
		panic("invalid mode value")
	}

	// Some pins don't support interrupts
	if p == PA08 {
		return
	}

	// Set up PMUX
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	pmux := int(p&0xFF) / 2
	if (p&0xFF)%2 == 0 {
		// Pin is odd numbered
		portgroup.Pmux[pmux].SetPmuxe(0)
	} else {
		portgroup.Pmux[pmux].SetPmuxo(0)
	}
	portgroup.Pincfg[p&0xFF].SetPmuxen(true)

	// Determine the EXTINT from the pin number
	exint := int((p & 0xFF) % 16)

	handlerFuncs[exint] = handler

	// Set the configuration
	config := exint / 8
	pos := exint * 4
	mask := (0x07 << (3 * (exint % 2))) << pos
	configVal := (mode << (3 * (exint % 2))) << pos

	eic.Eic.Config[config] = (eic.Eic.Config[config] & (^eic.RegisterConfigType(mask))) | eic.RegisterConfigType(configVal)

	// Enable the interrupt
	eic.Eic.Intenset |= 1 << exint

	// Enable EIC
	eic.Eic.Ctrl.SetEnable(true)
	for eic.Eic.Status.GetSyncbusy() {
	}

	// Enable the interrupt in NVIC
	irq := cortexm.Interrupt(12 + exint)
	// irq.SetPriority(0xC0)
	irq.EnableIRQ()
}

func (p Pin) SetPMUX(mode PMUXFunction, enabled bool) {
	// Set up PMUX
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	pmux := int(p&0xFF) / 2
	if (p&0xFF)%2 == 0 {
		// Pin is odd numbered
		portgroup.Pmux[pmux].SetPmuxe(port.ConstantPortPmuxPmuxeType(mode))
	} else {
		portgroup.Pmux[pmux].SetPmuxo(port.ConstantPortPmuxPmuxoType(mode))
	}
	portgroup.Pincfg[p&0xFF].SetPmuxen(enabled)
}

func (p Pin) ClearInterrupt() {
	// Determine the EXTINT from the pin number
	exint := (p & 0xFF) % 16
	if eic.Eic.Intenset&(1<<exint) != 0 {
		// Disable the interrupt
		eic.Eic.Intenclr &= 1 << exint

		// Disable PMUX
		portgroup := &port.Port.Group[p>>8]
		portgroup.Pincfg[p&0xFF].SetPmuxen(false)

		// Disable the interrupt in NVIC
		irq := cortexm.Interrupt(12 + exint)
		irq.DisableIRQ()

		handlerFuncs[exint] = nil
	}
}

func (p Pin) SetMode(mode Mode) {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	if mode == Input {
		portgroup.Dirclr.SetDirclr(1 << (p & 0xFF))
		portgroup.Ctrl.SetSampling(1 << (p & 0xFF))
	} else if mode == Output {
		portgroup.Dirset.SetDirset(1 << (p & 0xFF))
	}
	portgroup.Pincfg[p&0xFF].SetInen(true)
}

func (p Pin) GetMode() Mode {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.Dir.GetDir() == 0 {
		return Output
	}
	return Input
}

func (p Pin) SetPullMode(mode PullMode) {
	portgroup := &port.Port.Group[0xFF&(p>>8)]
	if (1<<(p&0xFF))&portgroup.Dir.GetDir() == 0 {
		if mode == PullDown {
			p.Set(false)
			portgroup.Pincfg[p&0xFF].SetPullen(true)
		} else if mode == PullUp {
			p.Set(true)
			portgroup.Pincfg[p&0xFF].SetPullen(true)
		} else { // NoPull
			portgroup.Pincfg[p&0xFF].SetPullen(false)
		}
	}
}

func (p Pin) GetPullMode() PullMode {
	return 0
}

func (p Pin) GetSERCOM() atsamd21.SERCOM {
	s := int(p>>28) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return atsamd21.SERCOM(s)
}

func (p Pin) GetAltSERCOM() atsamd21.SERCOM {
	s := int(p>>20) & 0x0F
	if s == 0x0F && p != 0 {
		return -1
	}
	return atsamd21.SERCOM(s)
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

//sigo:interrupt eicHandler EicHandler
func eicHandler() {
	status := volatile.LoadUint16((*uint16)(unsafe.Pointer(&eic.Eic.Intenset)))
	for n := range 16 {
		if (status>>n)&0x1 == 1 {
			if fn := handlerFuncs[n]; fn != nil {
				fn()
			}

			// Clear the interrupt flag
			volatile.StoreUint16((*uint16)(unsafe.Pointer(&eic.Eic.Intenset)), 1<<n)
		}
	}
}
