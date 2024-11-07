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

func f() int {
	return 0
}

func main() {
	var iii0 any = 0
	var iii1 any = 1
	var iiii int = 3

	c0 := iii0 == iii1
	c1 := iii0 == iiii
	use(c0)
	use(c1)
}

func use(v bool) {

}
