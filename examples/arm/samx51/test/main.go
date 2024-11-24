//go:build samx51

package main

// #include <stdio.h>
import "C"

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

}

func use(v any) {

}
