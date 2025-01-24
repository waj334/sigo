//go:build stm32h7x7

package main

import (
	"peripheral/i2c"
	"peripheral/pin"
	"peripheral/uart"
	"runtime/arm/cortexm/stm32/stm32h7x7"
)

const (
	// Timings for 120 MHz input clock.
	// NOTE: These values are provided by STM32CubeMX.

	// StandardTiming is 100 KHz.
	StandardTiming = 0x307075B1

	// FastModePlusTiming is 1 MHz.
	FastModePlusTiming = 0x301027FF

	LEDR = pin.PI12
	LEDB = pin.PE3
	LEDG = pin.PJ13

	ATECC608 = 0xC0
)

var (
	I2C  = i2c.I2C4
	UART = uart.UART1
)

func main() {
	// Initialize the clock system.
	stm32h7x7.DefaultClocks()

	LEDR.SetMode(pin.Output)
	LEDB.SetMode(pin.Output)
	LEDG.SetMode(pin.Output)

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

	// Configure the I2C host
	if err := I2C.Configure(i2c.Config{
		Enabled:        true,
		SDA:            pin.PH12,
		SCL:            pin.PB6,
		ClockFrequency: 100_000,
		Timing:         StandardTiming,
	}); err != nil {
		UART.WriteString(err.Error())
		UART.WriteString("\n")
		panic(err)
	}

	// ATECC608x wake sequence
	for {
		LEDR.High()
		LEDG.Low()
		LEDB.Low()

		I2C.SetTiming(100_000, StandardTiming)
		I2C.WriteAddress(0x00, []byte{0x00})
		I2C.SetTiming(1_000_000, FastModePlusTiming)

		wake := make([]byte, 4)
		if _, err := I2C.ReadAddress(ATECC608, wake); err != nil {
			UART.WriteString(err.Error())
			UART.WriteString("\n")
			continue
		}

		if wake[0] == 0x4 && wake[1] == 0x11 && wake[2] == 0x33 && wake[3] == 0x43 {
			UART.WriteString("ATECC608 Woke!\n")
			LEDR.Low()
			LEDG.High()
			LEDB.Low()
			return
		} else if wake[0] == 0x4 && wake[1] == 0x07 && wake[2] == 0xC4 && wake[3] == 0x40 {
			UART.WriteString("Self-test error!\n")
			LEDR.High()
			LEDG.Low()
			LEDB.High()
			return
		}
		UART.WriteString("retrying wake sequence\n")
	}
}
