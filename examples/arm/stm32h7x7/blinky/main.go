//go:build stm32h7x7

package main

import (
	"peripheral/pin"
	"runtime/arm/cortexm/stm32/stm32h7x7"
	"time"
)

const (
	LEDR = pin.PI12
	LEDB = pin.PE3
	LEDG = pin.PJ13
)

func init() {
	stm32h7x7.DefaultClocks()
}

func main() {
	LEDR.SetMode(pin.Output)
	LEDB.SetMode(pin.Output)
	LEDG.SetMode(pin.Output)

	for {
		LEDR.Toggle()
		time.Sleep(time.Millisecond * 500)
		LEDB.Toggle()
		time.Sleep(time.Millisecond * 500)
		LEDG.Toggle()
		time.Sleep(time.Millisecond * 500)
	}
}
