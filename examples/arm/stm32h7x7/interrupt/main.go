//go:build stm32h7x7

package main

import (
	"peripheral/pin"
	"runtime/arm/cortexm/stm32/stm32h7x7"
)

const (
	LEDR = pin.PI12
	LEDB = pin.PE3
	LEDG = pin.PJ13
	D2   = pin.PA3
)

func init() {
	stm32h7x7.DefaultClocks()
}

func main() {
	LEDR.SetMode(pin.Output)
	LEDB.SetMode(pin.Output)
	LEDG.SetMode(pin.Output)

	LEDR.Low()
	LEDG.Low()
	LEDB.Low()

	D2.SetMode(pin.Input)
	D2.SetInterrupt(pin.RisingEdge, func() {
		LEDR.Toggle()
	})

	// Block forever...
	select {}
}
