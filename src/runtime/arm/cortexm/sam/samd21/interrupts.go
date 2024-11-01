package samd21

import "runtime/arm/cortexm"

const (
	IRQ_PM cortexm.Interrupt = iota
	IRQ_SYSCTRL
	IRQ_WDT
	IRQ_RTC
	IRQ_EIC
	IRQ_NVMCTRL
	IRQ_DMAC
	IRQ_USB
	IRQ_EVSYS
	IRQ_SERCOM0
	IRQ_SERCOM1
	IRQ_SERCOM2
	IRQ_SERCOM3
	IRQ_SERCOM4
	IRQ_SERCOM5
	IRQ_TCC0
	IRQ_TCC1
	IRQ_TCC2
	IRQ_TC3
	IRQ_TC4
	IRQ_TC5
	IRQ_TC6
	IRQ_TC7
	IRQ_ADC
	IRQ_AC
	IRQ_DAC
	IRQ_PTC
	IRQ_I2S
	IRQ_AC1
	IRQ_TCC3
)
