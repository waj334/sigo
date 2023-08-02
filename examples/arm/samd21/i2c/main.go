//go:build samd21

package main

import (
	"peripheral/i2c"
	"peripheral/pin"
	"peripheral/uart"
	mcu "runtime/arm/cortexm/sam/samd21"
	"time"
)

var (
	I2C  = i2c.I2C3
	UART = uart.UART5
)

func main() {
	// Initialize the clock system
	mcu.DefaultClocks()

	// Configure UART
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

	// Configure the I2C host
	if err := I2C.Configure(i2c.Config{
		SDA:          pin.PA22,
		SCL:          pin.PA23,
		ClockSpeedHz: 100_000,
		MasterCode:   i2c.MasterCode1,
		Speed:        i2c.StandardAndFastMode,
	}); err != nil {
		UART.WriteString(err.Error() + "\n")
		panic(err)
	}

	addr := uint16(0xC0 >> 1)

	// ATECC608x wake sequence
	for {
		fSCL := I2C.GetClockFrequency()
		I2C.SetClockFrequency(100_000)
		I2C.WriteAddress(0x00, []byte{0x01})
		I2C.SetClockFrequency(fSCL)
		time.Sleep(time.Microsecond * 1500)

		wake := make([]byte, 4)
		if _, err := I2C.ReadAddress(addr, wake); err != nil {
			UART.WriteString(err.Error())
			UART.WriteString("\n")
			continue
		}

		if wake[0] == 0x4 && wake[1] == 0x11 && wake[2] == 0x33 && wake[3] == 0x43 {
			UART.WriteString("ATECC608 Woke!\n")
			return
		} else if wake[0] == 0x4 && wake[1] == 0x07 && wake[2] == 0xC4 && wake[3] == 0x40 {
			UART.WriteString("Self-test error!\n")
			return
		}
		UART.WriteString("retrying wake sequence\n")
	}
}
