//go:build samx51

package main

import (
	"peripheral/pin"
	"peripheral/spi"
	"peripheral/uart"
	mcu "runtime/arm/cortexm/sam/samx51"
)

var (
	UART = uart.UART5
	SPI  = spi.SPI0
)

func main() {
	// Initialize the clock system
	mcu.DefaultClocks()

	// Configure UART
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

	// Configure SPI
	if err := SPI.Configure(spi.Config{
		DI:             pin.PA04,
		DO:             pin.PA07,
		SCK:            pin.PA05,
		CS:             pin.PA06,
		BaudHz:         10_000_000,
		CharacterSize:  8,
		DataOrder:      spi.MSB,
		HardwareSelect: true,
		Phase:          spi.LeadingEdge,
		Polarity:       spi.IdleLow,
		ReceiveEnabled: true,
	}); err != nil {
		UART.WriteString(err.Error() + "\n")
		panic(err)
	}

	buf := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	UART.Write(buf)
	UART.WriteString("\n")
	SPI.Write(buf)
	SPI.Read(buf)
	UART.Write(buf)
	UART.WriteString("\n")
}
