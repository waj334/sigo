//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi
//sigo:features thumb2,soft-float
//sigo:float soft

package i2c

import (
	mcu "runtime/arm/cortexm/sam/atsamx51"
	i2c "runtime/arm/cortexm/sam/atsamx51/i2c/host"
	"runtime/arm/cortexm/sam/atsamx51/uart"
	_ "runtime/arm/cortexm/sam/chip/atsame51g19a"
	"time"
)

var (
	I2C  = i2c.I2C4
	UART = uart.UART5
)

func main() {
	// Initialize the FPU
	//mcu.InitFPU()

	// Initialize the clock system
	mcu.DefaultClocks()

	// Configure UART
	UART.Configure(uart.Config{
		TXD:             mcu.PB02,
		RXD:             mcu.PB03,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	// Configure the I2C host
	if err := I2C.Configure(i2c.Config{
		SDA:          mcu.PB08,
		SCL:          mcu.PB09,
		ClockSpeedHz: 100_000,
		MasterCode:   i2c.MasterCode1,
		Speed:        i2c.StandardAndFastMode,
	}); err != nil {
		UART.WriteString(err.Error() + "\n")
		panic(err)
	}

	// ATECC608x address is pre-shifted. Undo that
	address := uint16(0xC0 >> 1)

	// ATECC608x wake sequence
	for {
		fSCL := I2C.GetClockFrequency()
		I2C.SetClockFrequency(100_000)
		I2C.WriteAddress(0x00, []byte{0x01})
		I2C.SetClockFrequency(fSCL)
		time.Sleep(time.Microsecond * 1500)

		wake := make([]byte, 4)
		if _, err := I2C.ReadAddress(address, wake); err != nil {
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
