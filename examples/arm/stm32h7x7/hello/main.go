package main

import (
	"peripheral/pin"
	"peripheral/uart"
	"runtime/arm/cortexm/stm32/stm32h7x7"
)

var (
	UART = uart.UART1
)

func main() {
	stm32h7x7.DefaultClocks()

	// Configure UART
	UART.Configure(uart.Config{
		Enable: true,
		TX:     pin.PA9,
		RX:     pin.PB7,
		// FrameFormat:     uart.UsartFrame,
		Baud:            115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	UART.WriteString("Hello, World!\n\n")
	UART.WriteString("Any message received will be echoed below:\n")

	b := make([]byte, 32)
	for {
		n, err := UART.Read(b)
		if err != nil {
			panic(err)
		}

		if n > 0 {
			UART.Write(b[:n])
		}
	}
}
