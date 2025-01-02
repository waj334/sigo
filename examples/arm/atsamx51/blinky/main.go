//go:build atsamx5x

package main

import (
	"peripheral/pin"
	"runtime/arm/cortexm/sam/atsamx5x"
	"time"
)

var (
	LED = pin.PB11
)

func main() {
	// Initialize the clock system.
	atsamx5x.DefaultClocks()

	// Set up the LED.
	LED.SetDirection(pin.Output)
	LED.Set(true)

	// Blink forever.
	for {
		time.Sleep(time.Millisecond * 500)
		LED.Toggle()
	}
}
