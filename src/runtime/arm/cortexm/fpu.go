//go:build arm && fpu

package cortexm

import (
	"runtime/arm/cortexm/support/fpu"
	"runtime/arm/cortexm/support/systemcontrol"
)

type extendedFrame struct {
	Sn    [16]float32
	FPSCR uintptr
}

//sigo:extern _fpuEnabled runtime._fpuEnabled
var _fpuEnabled bool

func initFPU() {
	if _fpuEnabled {
		systemcontrol.SystemControl.Cpacr.SetCp10(systemcontrol.CpacrCp10Full)
		systemcontrol.SystemControl.Cpacr.SetCp11(systemcontrol.CpacrCp11Full)
		fpu.Fpu.Fpccr.SetAspen(true)
		fpu.Fpu.Fpccr.SetLspen(true)
	}
}
