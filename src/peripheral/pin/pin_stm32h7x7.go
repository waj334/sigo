//go:build stm32h7x7

package pin

import (
	"peripheral/adc"
	"peripheral/dac"
	"runtime/arm/cortexm/stm32/stm32h7x7"
	"runtime/arm/cortexm/stm32/stm32h7x7/support/exti"
	"runtime/arm/cortexm/stm32/stm32h7x7/support/gpio"
	"runtime/arm/cortexm/stm32/stm32h7x7/support/syscfg"
	"volatile"
)

const (
	pin00 = 0x0000_0000
	pin01 = 0x0000_0001
	pin02 = 0x0000_0002
	pin03 = 0x0000_0003
	pin04 = 0x0000_0004
	pin05 = 0x0000_0005
	pin06 = 0x0000_0006
	pin07 = 0x0000_0007
	pin08 = 0x0000_0008
	pin09 = 0x0000_0009
	pin10 = 0x0000_000A
	pin11 = 0x0000_000B
	pin12 = 0x0000_000C
	pin13 = 0x0000_000D
	pin14 = 0x0000_000E
	pin15 = 0x0000_000F

	gpioA = 0x0000_0000
	gpioB = 0x0000_0010
	gpioC = 0x0000_0020
	gpioD = 0x0000_0030
	gpioE = 0x0000_0040
	gpioF = 0x0000_0050
	gpioG = 0x0000_0060
	gpioH = 0x0000_0070
	gpioI = 0x0000_0080
	gpioJ = 0x0000_0090
	gpioK = 0x0000_00A0

	adc1    = 0x0000_0100
	adc2    = 0x0000_0200
	adc3    = 0x0000_0300
	dacOut1 = 0x0000_0400
	dacOut2 = 0x0000_0800

	adcInp0  = 0x0000_1000
	adcInp1  = 0x0000_2000
	adcInp2  = 0x0000_3000
	adcInp3  = 0x0000_4000
	adcInp4  = 0x0000_5000
	adcInp5  = 0x0000_6000
	adcInp6  = 0x0000_7000
	adcInp7  = 0x0000_8000
	adcInp8  = 0x0000_9000
	adcInp9  = 0x0000_a000
	adcInp10 = 0x0000_b000
	adcInp11 = 0x0000_c000
	adcInp12 = 0x0000_d000
	adcInp13 = 0x0000_e000
	adcInp14 = 0x0000_f000
	adcInp15 = 0x0001_0000
	adcInp16 = 0x0001_1000
	adcInp17 = 0x0001_2000
	adcInp18 = 0x0001_3000
	adcInp19 = 0x0001_4000

	adcInn0  = 0x0010_0000
	adcInn1  = 0x0020_0000
	adcInn2  = 0x0030_0000
	adcInn3  = 0x0040_0000
	adcInn4  = 0x0050_0000
	adcInn5  = 0x0060_0000
	adcInn6  = 0x0070_0000
	adcInn7  = 0x0080_0000
	adcInn8  = 0x0090_0000
	adcInn9  = 0x00a0_0000
	adcInn10 = 0x00b0_0000
	adcInn11 = 0x00c0_0000
	adcInn12 = 0x00d0_0000
	adcInn13 = 0x00e0_0000
	adcInn14 = 0x00f0_0000
	adcInn15 = 0x0100_0000
	adcInn16 = 0x0110_0000
	adcInn17 = 0x0120_0000
	adcInn18 = 0x0130_0000
	adcInn19 = 0x0140_0000

	Input       Mode = 0x0
	Output      Mode = 0x1
	AltFunction Mode = 0x2
	Analog      Mode = 0x3

	Alt0  Mode = 0x00
	Alt1  Mode = 0x10
	Alt2  Mode = 0x20
	Alt3  Mode = 0x30
	Alt4  Mode = 0x40
	Alt5  Mode = 0x50
	Alt6  Mode = 0x60
	Alt7  Mode = 0x70
	Alt8  Mode = 0x80
	Alt9  Mode = 0x90
	Alt10 Mode = 0xA0
	Alt11 Mode = 0xB0
	Alt12 Mode = 0xC0
	Alt13 Mode = 0xD0
	Alt14 Mode = 0xE0
	Alt15 Mode = 0xF0

	NoEdge      IRQMode = 0
	RisingEdge  IRQMode = 1
	FallingEdge IRQMode = 2
	BothEdges   IRQMode = 3

	NoPull   PullMode = 0
	PullUp   PullMode = 1
	PullDown PullMode = 2

	PushPull  OutputMode = 0
	OpenDrain OutputMode = 1

	LowSpeed      SpeedMode = 0
	MediumSpeed   SpeedMode = 1
	HighSpeed     SpeedMode = 2
	VeryHighSpeed SpeedMode = 3
)

// GPIO group A
const (
	PA0  Pin = pin00 | gpioA | adc1 | adcInp16
	PA1  Pin = pin01 | gpioA | adc1 | adcInp17 | adcInn16
	PA2  Pin = pin02 | gpioA | adc1 | adc2 | adcInp14
	PA3  Pin = pin03 | gpioA | adc1 | adc2 | adcInp15
	PA4  Pin = pin04 | gpioA | adc1 | adc2 | dacOut1 | adcInp18
	PA5  Pin = pin05 | gpioA | adc1 | adc2 | dacOut2 | adcInp19 | adcInn18
	PA6  Pin = pin06 | gpioA | adc1 | adc2 | adcInp3
	PA7  Pin = pin07 | gpioA | adc1 | adc2 | adcInp7 | adcInn3
	PA8  Pin = pin08 | gpioA
	PA9  Pin = pin09 | gpioA
	PA10 Pin = pin10 | gpioA
	PA11 Pin = pin11 | gpioA
	PA12 Pin = pin12 | gpioA
	PA13 Pin = pin13 | gpioA
	PA14 Pin = pin14 | gpioA
	PA15 Pin = pin15 | gpioA
)

// GPIO group B
const (
	PB0  Pin = pin00 | gpioB | adc1 | adc2 | adcInp9 | adcInn5
	PB1  Pin = pin01 | gpioB | adc1 | adc2 | adcInp5
	PB2  Pin = pin02 | gpioB
	PB3  Pin = pin03 | gpioB
	PB4  Pin = pin04 | gpioB
	PB5  Pin = pin05 | gpioB
	PB6  Pin = pin06 | gpioB
	PB7  Pin = pin07 | gpioB
	PB8  Pin = pin08 | gpioB
	PB9  Pin = pin09 | gpioB
	PB10 Pin = pin10 | gpioB
	PB11 Pin = pin11 | gpioB
	PB12 Pin = pin12 | gpioB
	PB13 Pin = pin13 | gpioB
	PB14 Pin = pin14 | gpioB
	PB15 Pin = pin15 | gpioB
)

// GPIO group C
const (
	PC0  Pin = pin00 | gpioC | adc1 | adc2 | adc3 | adcInp10
	PC1  Pin = pin01 | gpioC | adc1 | adc2 | adc3 | adcInp11 | adcInn10
	PC2  Pin = pin02 | gpioC | adc1 | adc2 | adc3 | adcInp12 | adcInn11
	PC3  Pin = pin03 | gpioC | adc1 | adc2 | adcInp13 | adcInn12
	PC4  Pin = pin04 | gpioC | adc1 | adc2 | adcInp4
	PC5  Pin = pin05 | gpioC | adc1 | adc2 | adcInp8 | adcInn4
	PC6  Pin = pin06 | gpioC
	PC7  Pin = pin07 | gpioC
	PC8  Pin = pin08 | gpioC
	PC9  Pin = pin09 | gpioC
	PC10 Pin = pin10 | gpioC
	PC11 Pin = pin11 | gpioC
	PC12 Pin = pin12 | gpioC
	PC13 Pin = pin13 | gpioC
	PC14 Pin = pin14 | gpioC
	PC15 Pin = pin15 | gpioC
)

// GPIO group D
const (
	PD0  Pin = pin00 | gpioD
	PD1  Pin = pin01 | gpioD
	PD2  Pin = pin02 | gpioD
	PD3  Pin = pin03 | gpioD
	PD4  Pin = pin04 | gpioD
	PD5  Pin = pin05 | gpioD
	PD6  Pin = pin06 | gpioD
	PD7  Pin = pin07 | gpioD
	PD8  Pin = pin08 | gpioD
	PD9  Pin = pin09 | gpioD
	PD10 Pin = pin10 | gpioD
	PD11 Pin = pin11 | gpioD
	PD12 Pin = pin12 | gpioD
	PD13 Pin = pin13 | gpioD
	PD14 Pin = pin14 | gpioD
	PD15 Pin = pin15 | gpioD
)

// GPIO group E
const (
	PE0  Pin = pin00 | gpioE
	PE1  Pin = pin01 | gpioE
	PE2  Pin = pin02 | gpioE
	PE3  Pin = pin03 | gpioE
	PE4  Pin = pin04 | gpioE
	PE5  Pin = pin05 | gpioE
	PE6  Pin = pin06 | gpioE
	PE7  Pin = pin07 | gpioE
	PE8  Pin = pin08 | gpioE
	PE9  Pin = pin09 | gpioE
	PE10 Pin = pin10 | gpioE
	PE11 Pin = pin11 | gpioE
	PE12 Pin = pin12 | gpioE
	PE13 Pin = pin13 | gpioE
	PE14 Pin = pin14 | gpioE
	PE15 Pin = pin15 | gpioE
)

// GPIO group F
const (
	PF0  Pin = pin00 | gpioF
	PF1  Pin = pin01 | gpioF
	PF2  Pin = pin02 | gpioF
	PF3  Pin = pin03 | gpioF | adc3 | adcInp5
	PF4  Pin = pin04 | gpioF | adc3 | adcInp9 | adcInn5
	PF5  Pin = pin05 | gpioF | adc3 | adcInp4
	PF6  Pin = pin06 | gpioF | adc3 | adcInp8 | adcInn4
	PF7  Pin = pin07 | gpioF | adc3 | adcInp9
	PF8  Pin = pin08 | gpioF | adc3 | adcInp7 | adcInn3
	PF9  Pin = pin09 | gpioF | adc3 | adcInp2
	PF10 Pin = pin10 | gpioF | adc3 | adcInp6 | adcInn2
	PF11 Pin = pin11 | gpioF | adc1 | adcInp2
	PF12 Pin = pin12 | gpioF | adc1 | adcInp6 | adcInn2
	PF13 Pin = pin13 | gpioF | adc2 | adcInp2
	PF14 Pin = pin14 | gpioF | adc2 | adcInp6 | adcInn2
	PF15 Pin = pin15 | gpioF
)

// GPIO group G
const (
	PG0  Pin = pin00 | gpioG
	PG1  Pin = pin01 | gpioG
	PG2  Pin = pin02 | gpioG
	PG3  Pin = pin03 | gpioG
	PG4  Pin = pin04 | gpioG
	PG5  Pin = pin05 | gpioG
	PG6  Pin = pin06 | gpioG
	PG7  Pin = pin07 | gpioG
	PG8  Pin = pin08 | gpioG
	PG9  Pin = pin09 | gpioG
	PG10 Pin = pin10 | gpioG
	PG11 Pin = pin11 | gpioG
	PG12 Pin = pin12 | gpioG
	PG13 Pin = pin13 | gpioG
	PG14 Pin = pin14 | gpioG
	PG15 Pin = pin15 | gpioG
)

// GPIO group H
const (
	PH0  Pin = pin00 | gpioH
	PH1  Pin = pin01 | gpioH
	PH2  Pin = pin02 | gpioH | adc3 | adcInp13
	PH3  Pin = pin03 | gpioH | adc3 | adcInp14 | adcInn13
	PH4  Pin = pin04 | gpioH | adc3 | adcInp15 | adcInn14
	PH5  Pin = pin05 | gpioH | adc3 | adcInp16 | adcInn15
	PH6  Pin = pin06 | gpioH
	PH7  Pin = pin07 | gpioH
	PH8  Pin = pin08 | gpioH
	PH9  Pin = pin09 | gpioH
	PH10 Pin = pin10 | gpioH
	PH11 Pin = pin11 | gpioH
	PH12 Pin = pin12 | gpioH
	PH13 Pin = pin13 | gpioH
	PH14 Pin = pin14 | gpioH
	PH15 Pin = pin15 | gpioH
)

// GPIO group I
const (
	PI0  Pin = pin00 | gpioI
	PI1  Pin = pin01 | gpioI
	PI2  Pin = pin02 | gpioI
	PI3  Pin = pin03 | gpioI
	PI4  Pin = pin04 | gpioI
	PI5  Pin = pin05 | gpioI
	PI6  Pin = pin06 | gpioI
	PI7  Pin = pin07 | gpioI
	PI8  Pin = pin08 | gpioI
	PI9  Pin = pin09 | gpioI
	PI10 Pin = pin10 | gpioI
	PI11 Pin = pin11 | gpioI
	PI12 Pin = pin12 | gpioI
	PI13 Pin = pin13 | gpioI
	PI14 Pin = pin14 | gpioI
	PI15 Pin = pin15 | gpioI
)

// GPIO group J
const (
	PJ0  Pin = pin00 | gpioJ
	PJ1  Pin = pin01 | gpioJ
	PJ2  Pin = pin02 | gpioJ
	PJ3  Pin = pin03 | gpioJ
	PJ4  Pin = pin04 | gpioJ
	PJ5  Pin = pin05 | gpioJ
	PJ6  Pin = pin06 | gpioJ
	PJ7  Pin = pin07 | gpioJ
	PJ8  Pin = pin08 | gpioJ
	PJ9  Pin = pin09 | gpioJ
	PJ10 Pin = pin10 | gpioJ
	PJ11 Pin = pin11 | gpioJ
	PJ12 Pin = pin12 | gpioJ
	PJ13 Pin = pin13 | gpioJ
	PJ14 Pin = pin14 | gpioJ
	PJ15 Pin = pin15 | gpioJ
)

// GPIO group K
const (
	PK0  Pin = pin00 | gpioK
	PK1  Pin = pin01 | gpioK
	PK2  Pin = pin02 | gpioK
	PK3  Pin = pin03 | gpioK
	PK4  Pin = pin04 | gpioK
	PK5  Pin = pin05 | gpioK
	PK6  Pin = pin06 | gpioK
	PK7  Pin = pin07 | gpioK
	PK8  Pin = pin08 | gpioK
	PK9  Pin = pin09 | gpioK
	PK10 Pin = pin10 | gpioK
	PK11 Pin = pin11 | gpioK
	PK12 Pin = pin12 | gpioK
	PK13 Pin = pin13 | gpioK
	PK14 Pin = pin14 | gpioK
	PK15 Pin = pin15 | gpioK
)

var (
	handlerFuncs [16]func()
)

type Pin uint32

func (p Pin) number() int {
	return int(p) & 0xF
}

func (p Pin) group() int {
	return int(p>>4) & 0xF
}

func (p Pin) adc() int {
	return int(p>>8) & 0xF
}

func (p Pin) adcP() int {
	return (int(p>>12) & 0xFF) - 1
}

func (p Pin) adcN() int {
	return (int(p>>20) & 0xFF) - 1
}

func (p Pin) dacN() int {
	return int(p>>10) & 0x3
}

func (p Pin) High() {
	p.Set(true)
}

func (p Pin) Low() {
	p.Set(false)
}

func (p Pin) Toggle() {
	p.Set(p.Get())
}

func (p Pin) Set(on bool) {
	group := gpio.Gpio[p.group()]
	n := p.number()
	if on {
		volatile.StoreUint32((*uint32)(&group.Bsrr), 1<<(16+n))
	} else {
		volatile.StoreUint32((*uint32)(&group.Bsrr), 1<<n)
	}
}

func (p Pin) Get() bool {
	group := gpio.Gpio[p.group()]
	n := p.number()
	switch p.GetMode() {
	case Input:
		return volatile.LoadUint32((*uint32)(&group.Idr))&(1<<n) != 0
	case Output:
		return volatile.LoadUint32((*uint32)(&group.Odr))&(1<<n) != 0
	default:
		panic("unreachable")
	}
}

func (p Pin) SetValue(value uint) error {
	var err error
	switch p.GetMode() {
	case Input, Output:
		p.Set(value != 0)
	case Analog:
		if dacN := p.dacN(); dacN > 0 {
			switch dacN {
			case 1:
				err = dac.DAC1.Write(value)
			case 2:
				err = dac.DAC2.Write(value)
			}
		}
	default:
		panic("invalid pin mode")
	}
	return err
}

func (p Pin) Value() (value uint, err error) {
	switch {
	case p&adc1 != 0:
		// TODO: Check the ADC mode and fallthrough to ADC2 if this one is single conversion and busy.
		value, err = adc.ADC1.Read(p.adcP())
	case p&adc2 != 0:
		value, err = adc.ADC2.Read(p.adcP())
	case p&adc3 != 0:
		value, err = adc.ADC3.Read(p.adcP())
	}
	return
}

func (p Pin) SetInterrupt(mode IRQMode, handler func()) {
	group := uint8(p.group())
	pin := p.number()
	mask := uint32(1 << pin)

	// Set the pin group for the EXTI corresponding to the pin number.
	// NOTE: STM32H series does NOT support interrupts on multiple pins sharing the same number in different groups. For
	//       example, PA0 and PB0 cannot share the same interrupt at the same time.
	switch pin {
	case 0:
		syscfg.Syscfg.Exticr1.SetExti0(group)
	case 1:
		syscfg.Syscfg.Exticr1.SetExti1(group)
	case 2:
		syscfg.Syscfg.Exticr1.SetExti2(group)
	case 3:
		syscfg.Syscfg.Exticr1.SetExti3(group)
	case 4:
		syscfg.Syscfg.Exticr2.SetExti4(group)
	case 5:
		syscfg.Syscfg.Exticr2.SetExti5(group)
	case 6:
		syscfg.Syscfg.Exticr2.SetExti6(group)
	case 7:
		syscfg.Syscfg.Exticr2.SetExti7(group)
	case 8:
		syscfg.Syscfg.Exticr3.SetExti8(group)
	case 9:
		syscfg.Syscfg.Exticr3.SetExti9(group)
	case 10:
		syscfg.Syscfg.Exticr3.SetExti10(group)
	case 11:
		syscfg.Syscfg.Exticr3.SetExti11(group)
	case 12:
		syscfg.Syscfg.Exticr4.SetExti12(group)
	case 13:
		syscfg.Syscfg.Exticr4.SetExti13(group)
	case 14:
		syscfg.Syscfg.Exticr4.SetExti14(group)
	case 15:
		syscfg.Syscfg.Exticr4.SetExti15(group)
	}

	// Unmask the interrupt (Enable EXTI for the pin)
	volatile.StoreUint32((*uint32)(&exti.Exti.Cpuimr1), volatile.LoadUint32((*uint32)(&exti.Exti.Cpuimr1))|mask)

	// Clear trigger settings
	volatile.StoreUint32((*uint32)(&exti.Exti.Rtsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Rtsr1))&^mask)
	volatile.StoreUint32((*uint32)(&exti.Exti.Ftsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Ftsr1))&^mask)

	// Configure Edge Trigger.
	if mode == RisingEdge || mode == BothEdges {
		volatile.StoreUint32((*uint32)(&exti.Exti.Rtsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Rtsr1))|mask)
	}

	if mode == FallingEdge || mode == BothEdges {
		volatile.StoreUint32((*uint32)(&exti.Exti.Ftsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Ftsr1))|mask)
	}

	// Enable IRQ in NVIC.
	switch {
	case pin == 0:
		stm32h7x7.IrqExti0.EnableIRQ()
		stm32h7x7.IrqExti0.SetPriority(1)
	case pin == 1:
		stm32h7x7.IrqExti1.EnableIRQ()
		stm32h7x7.IrqExti1.SetPriority(1)
	case pin == 2:
		stm32h7x7.IrqExti2.EnableIRQ()
		stm32h7x7.IrqExti2.SetPriority(1)
	case pin == 3:
		stm32h7x7.IrqExti3.EnableIRQ()
		stm32h7x7.IrqExti3.SetPriority(1)
	case pin >= 4 && pin <= 9:
		stm32h7x7.IrqExti95.EnableIRQ()
		stm32h7x7.IrqExti95.SetPriority(1)
	case pin >= 10 && pin <= 15:
		stm32h7x7.IrqExti1510.EnableIRQ()
		stm32h7x7.IrqExti1510.SetPriority(1)
	default:
		panic("unreachable")
	}

	// Replace the handler for this pin number.
	handlerFuncs[pin] = handler
}

func (p Pin) ClearInterrupt() {
	pin := p.number()
	mask := uint32(1 << pin)

	// Disable EXTI interrupt
	volatile.StoreUint32((*uint32)(&exti.Exti.Cpuimr1), volatile.LoadUint32((*uint32)(&exti.Exti.Cpuimr1))&^mask)

	// Disable rising/falling edge triggers
	volatile.StoreUint32((*uint32)(&exti.Exti.Rtsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Rtsr1))&^mask)
	volatile.StoreUint32((*uint32)(&exti.Exti.Ftsr1), volatile.LoadUint32((*uint32)(&exti.Exti.Ftsr1))&^mask)

	// Enable IRQ in NVIC.
	switch {
	case pin == 0:
		stm32h7x7.IrqExti0.DisableIRQ()
	case pin == 1:
		stm32h7x7.IrqExti1.DisableIRQ()
	case pin == 2:
		stm32h7x7.IrqExti2.DisableIRQ()
	case pin == 3:
		stm32h7x7.IrqExti3.DisableIRQ()
	case pin >= 4 && pin <= 9:
		stm32h7x7.IrqExti95.DisableIRQ()
	case pin >= 10 && pin <= 15:
		stm32h7x7.IrqExti1510.DisableIRQ()
	default:
		panic("unreachable")
	}

	// Unset the handler for this pin number.
	handlerFuncs[pin] = nil
}

func (p Pin) SetMode(mode Mode) {
	pinMode := mode & 0x3
	alt := (mode >> 4) & 0xF

	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	volatile.StoreUint32((*uint32)(&group.Moder),
		(volatile.LoadUint32((*uint32)(&group.Moder))&^mask)|(uint32(pinMode)<<offsetInBits))

	if pinMode != AltFunction {
		alt = 0
	}

	// NOTE: The low register covers pins [0-7] and the high register covers pins [8-15]
	offsetInBits = uint32(4 * (n % 8))
	mask = uint32(0b1111) << offsetInBits
	if n < 8 {
		// Set the alternate function low register.
		volatile.StoreUint32((*uint32)(&group.Afrl),
			(volatile.LoadUint32((*uint32)(&group.Afrl))&^mask)|(uint32(alt)<<offsetInBits))
	} else {
		// Set the alternate function high register.
		volatile.StoreUint32((*uint32)(&group.Afrh),
			(volatile.LoadUint32((*uint32)(&group.Afrh))&^mask)|(uint32(alt)<<offsetInBits))
	}
}

func (p Pin) GetMode() Mode {
	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	return Mode((volatile.LoadUint32((*uint32)(&group.Moder)) & mask) >> offsetInBits)
}

func (p Pin) SetOutputMode(mode OutputMode) {
	group := gpio.Gpio[p.group()]
	n := p.number()
	mask := uint32(0x1) << n
	volatile.StoreUint32((*uint32)(&group.Otyper),
		(volatile.LoadUint32((*uint32)(&group.Otyper))&^mask)|(uint32(mode)<<n))
}

func (p Pin) GetOutputMode() OutputMode {
	group := gpio.Gpio[p.group()]
	n := p.number()
	mask := uint32(0x1) << n
	return OutputMode((volatile.LoadUint32((*uint32)(&group.Otyper)) & mask) >> n)
}

func (p Pin) SetSpeedMode(speed SpeedMode) {
	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	volatile.StoreUint32((*uint32)(&group.Ospeedr),
		(volatile.LoadUint32((*uint32)(&group.Ospeedr))&^mask)|(uint32(speed)<<offsetInBits))
}

func (p Pin) GetSpeedMode() SpeedMode {
	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	return SpeedMode((volatile.LoadUint32((*uint32)(&group.Ospeedr)) & mask) >> offsetInBits)
}

func (p Pin) SetPullMode(mode PullMode) {
	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	volatile.StoreUint32((*uint32)(&group.Pupdr),
		(volatile.LoadUint32((*uint32)(&group.Pupdr))&^mask)|(uint32(mode)<<offsetInBits))
}

func (p Pin) GetPullMode() PullMode {
	group := gpio.Gpio[p.group()]
	n := p.number()
	offsetInBits := uint32(2 * n)
	mask := uint32(0x3) << offsetInBits
	return PullMode((volatile.LoadUint32((*uint32)(&group.Pupdr)) & mask) >> offsetInBits)
}

//sigo:interrupt exti0Handler Exti0Handler
func exti0Handler() {
	extiHandler(0)
}

//sigo:interrupt exti1Handler Exti1Handler
func exti1Handler() {
	extiHandler(1)
}

//sigo:interrupt exti2Handler Exti2Handler
func exti2Handler() {
	extiHandler(2)
}

//sigo:interrupt exti3Handler Exti3Handler
func exti3Handler() {
	extiHandler(3)
}

//sigo:interrupt exti4Handler Exti4Handler
func exti4Handler() {
	extiHandler(4)
}

//sigo:interrupt exti95Handler Exti95Handler
func exti95Handler() {
	// Read the pending register once to avoid multiple memory accesses
	pending := volatile.LoadUint32((*uint32)(&exti.Exti.Cpupr1)) & 0b1111100000 // Mask EXTI5-9

	// Iterate over each pending EXTI line and handle it
	for i := 5; i <= 9; i++ {
		if pending&(1<<i) != 0 { // Check if EXTI line i is pending
			extiHandler(i) // Call the handler for this EXTI line
		}
	}
}

//sigo:interrupt exti1510Handler Exti1510Handler
func exti1510Handler() {
	// Read the pending register once to avoid multiple memory accesses
	pending := volatile.LoadUint32((*uint32)(&exti.Exti.Cpupr1)) & 0b1111110000000000 // Mask EXTI10-15

	// Iterate over each pending EXTI line and handle it
	for i := 10; i <= 15; i++ {
		if pending&(1<<i) != 0 { // Check if EXTI line i is pending
			extiHandler(i) // Call the handler for this EXTI line
		}
	}
}

func extiHandler(i int) {
	if f := handlerFuncs[i]; f != nil {
		f()
	}
	// Clear the pending interrupt flag.
	volatile.StoreUint32((*uint32)(&exti.Exti.Cpupr1), 1<<i)
}
