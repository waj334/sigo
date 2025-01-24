//go:build arm && fpu

package fpu

// NOTE: Add `--float hardfp` to build command for target CPUs that support floating-point instructions.
//       Otherwise, no floating-point instruction will be emitted into the binary.

func main() {
	go func() {
		f1 := float32(0.5)
		f2 := float32(0.55)
		for {
			f3 := f1 + f2
			f1 += f3
			f2 -= f1
		}
	}()

	go func() {
		f4 := 0.6
		f5 := 0.4
		for {
			f6 := f4 + f5
			f4 += f6
			f5 -= f4
		}
	}()

	select {}
}

func use(any) {
	// Does nothing.
}
