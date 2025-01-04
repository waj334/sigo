//go:build arm && !fpu

package cortexm

func initFPU() {
	// Do nothing.
}
