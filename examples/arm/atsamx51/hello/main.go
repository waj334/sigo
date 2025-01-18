//go:build atsamx5x

package main

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamx5x"
	"time"
)

func main() {
	// Initialize the clock system.
	atsamx5x.DefaultClocks()

	cortexm.Semihosting.WriteString("Hello World\n")
	var input [128]byte
	for {
		count, err := cortexm.Semihosting.Read(input[:127])
		if err != nil {
			panic(err)
		}

		if count > 0 {
			// Echo back the input.
			cortexm.Semihosting.Write(input[:count+1])
			time.Sleep(time.Second)
		}
	}
}
