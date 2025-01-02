//go:build atsamd21

package main

import (
	"time"

	"peripheral/pin"
	"peripheral/uart"

	"runtime/arm/cortexm/sam/atsamd21"
)

var (
	UART = uart.UART5
)

func main() {
	atsamd21.DefaultClocks()
	UART.Configure(uart.Config{
		TXD:             pin.PB22,
		RXD:             pin.PB23,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	for {
		UART.WriteString("hello\n")
		time.Sleep(time.Second)
	}
}
