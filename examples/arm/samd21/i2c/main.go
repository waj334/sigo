//sigo:architecture arm
//sigo:cpu cortex-m0
//sigo:triple armv6m-none-eabi
//sigo:features

package i2c

import (
	mcu "runtime/arm/cortexm/sam/atsamd21"
	i2c "runtime/arm/cortexm/sam/atsamd21/i2c/host"
	"runtime/arm/cortexm/sam/atsamd21/uart"
	_ "runtime/arm/cortexm/sam/chip/atsamd21g18a"
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
		TXD:             mcu.PB22,
		RXD:             mcu.PB23,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	// Configure the I2C host
	if err := I2C.Configure(i2c.Config{
		SDA:          mcu.PA22,
		SCL:          mcu.PA23,
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
