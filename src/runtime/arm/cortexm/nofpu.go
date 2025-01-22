//go:build arm && !fpu

package cortexm

type extendedFrame struct {
	// This part of the stack stackFrame should not be considered when the FPU is disabled.
}

func initFPU() {
	// Do nothing.
}
