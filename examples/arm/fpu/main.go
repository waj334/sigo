//go:build arm && fpu

package fpu

// NOTE: Add `--float hardfp` to build command for target CPUs that support floating-point instructions.
//       Otherwise, no floating-point instruction will be emitted into the binary.

func main() {
	f1 := float32(0.5)
	f2 := float32(0.55)
	f3 := f1 + f2
	use(f3)
}

func use(any) {
	// Does nothing.
}
