//go:build thumb2

package cortexm

import (
	"asm"
	"asm/register"
)

//sigo:interrupt hardfaultHandler HardfaultHandler
func hardfaultHandler() {
	var es *stackFrame
	asm.Inline(`
		tst lr, #4
		ite eq
		mrseq {{es}}, msp
		mrsne {{es}}, psp
	`, asm.Out(&es), asm.Clobber(register.LR))
	hardfault(es)
}
