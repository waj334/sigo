//go:build stm32h7x7

package stm32h7x7

import (
	"runtime/arm/cortexm"
)

func init() {
	// The default clock speed for the Cortex-M7 core is 100MHz.
	cortexm.SysTickFrequency = 100_000_000

	// The Cortex-M7 core has 8 priority bits for interrupts.
	cortexm.IrqPriorityMask = 0xFF
}
