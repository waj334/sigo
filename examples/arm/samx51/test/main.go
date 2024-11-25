//go:build samx51

package main

import (
	"peripheral/pin"
	"peripheral/uart"
	mcu "runtime/arm/cortexm/sam/samx51"
)

var (
	UART = uart.UART5
	LED  = pin.PB11
)

func init() {
	// Initialize the clock system.
	mcu.DefaultClocks()

	// Configure UART.
	UART.Configure(uart.Config{
		TXD:             pin.PB02,
		RXD:             pin.PB03,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	// Set up the LED.
	LED.SetDirection(pin.Output)
	LED.Set(true)
}

func main() {
	s0 := [4]int{0, 1, 2, 3}
	//s0 := "test"

	var i int
	//var e rune
	var e int

	for _, _ = range s0 {
	}

	for i = range s0 {
		use(i)
	}

	for i, e = range s0 {
		use(i)
		use(e)
	}

	for i, e = range s0 {
		use(i)
		continue
		use(e)
	}

	for i, e = range s0 {
		break
		use(i)
		use(e)
	}

	for ii := range s0 {
		use(ii)
	}

	for iii, ee := range s0 {
		use(iii)
		use(ee)
	}
}

func use(v any) {

}
