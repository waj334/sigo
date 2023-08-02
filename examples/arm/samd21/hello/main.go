//go:build samd21

package main

import (
	"peripheral/pin"
	"peripheral/uart"
	mcu "runtime/arm/cortexm/sam/samd21"
	"time"
)

var (
	UART = uart.UART5
)

func initMCU() {
	mcu.DefaultClocks()
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
}

func main() {
	initMCU()
	for {
		UART.WriteString("hello\n")
		time.Sleep(time.Second)
	}
}
