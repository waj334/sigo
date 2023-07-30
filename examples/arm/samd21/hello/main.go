//sigo:architecture arm
//sigo:cpu cortex-m0
//sigo:triple armv6m-none-eabi
//sigo:features

package main

import (
	"time"

	mcu "runtime/arm/cortexm/sam/atsamd21"
	"runtime/arm/cortexm/sam/atsamd21/uart"
	_ "runtime/arm/cortexm/sam/chip/atsamd21g18a"
)

var (
	UART = uart.UART5
)

func initMCU() {
	mcu.DefaultClocks()
	UART.Configure(uart.Config{
		TXD:             mcu.PB22,
		RXD:             mcu.PB23,
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
