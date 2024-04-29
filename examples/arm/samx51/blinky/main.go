//go:build samx51

package main

import (
	"peripheral/pin"
	mcu "runtime/arm/cortexm/sam/samx51"
	"time"
)

var (
	LED = pin.PB11
)

func main() {
	// Initialize the clock system
	mcu.DefaultClocks()

	// Set up the LED
	LED.SetDirection(pin.Output)
	LED.Set(true)

	// Blink forever
	for {
		time.Sleep(time.Millisecond * 500)
		LED.Toggle()
	}
}
