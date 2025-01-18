//go:build !thumb2

package cortexm

import (
	"asm"
)

//sigo:interrupt hardfaultHandler HardfaultHandler
func hardfaultHandler() {
	var es *stackFrame
	asm.Inline(`
    mov r1, lr
    lsrs r1, #2
    bcc use_msp
    mrs {es}, psp

use_msp:
	mrs {es}, msp
	`, asm.Out(&es), asm.Clobber(registers.R1))
	hardfault(es)
}
