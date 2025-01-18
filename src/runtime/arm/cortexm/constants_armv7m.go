//go:build armv7m

package cortexm

const (
	defaultPsrValue uintptr = 0x0100_0000 // defaultPsrValue On armv7-m, the thumb bit MUST be set!
)
