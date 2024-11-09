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
	s0 := []int{0, 1, 2, 3}
	s1 := s0[1:]
	s2 := s0[:1]
	s3 := s0[0:2]
	s4 := s0[0:2:3]
	s5 := s0[:]

	str0 := "test"
	str1 := str0[1:]
	str2 := str0[:1]
	str3 := str0[0:2]
	str4 := str0[:]

	e0 := s0[2]
	e1 := str0[2]

	ptr0 := &s0[1]

	use(s1)
	use(s2)
	use(s3)
	use(s4)
	use(s5)
	use(str1)
	use(str2)
	use(str3)
	use(str4)
	use(e0)
	use(e1)
	use(ptr0)
}

func use(v any) {

}
