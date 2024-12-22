//go:build qemu_cortexm4

package cortexm4

import (
	"runtime/arm/cortexm"
)

// static const int usart_irq[] = { 37, 38, 39, 52, 53, 71, 82, 83 };

const (
	IRQ_USART1 cortexm.Interrupt = iota + 37
	IRQ_USART2
	IRQ_USART3
	IRQ_USART4 = iota + 49
	IRQ_USART5
	IRQ_USART6 = iota + 66
	IRQ_USART7 = iota + 76
	IRQ_USART8
)
