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
	var a []int
	var c, c1, c2, c3, c4 chan int
	var i1, i2 int
	select {
	case i1 = <-c1:
		use(i1)
	case c2 <- i2:
	case i3, ok := (<-c3): // same as: i3, ok := <-c3
		if ok {
			use(i3)
		} else {

		}
	case a[f()] = <-c4:
		// same as:
		// case t := <-c4
		//	a[f()] = t
	default:

	}

	for { // send random sequence of bits to c
		select {
		case c <- 0: // note: no statement, no fallthrough, no folding of cases
		case c <- 1:
		}
	}

	select {} // block forever
}

func use(v int) {

}
