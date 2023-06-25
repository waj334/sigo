//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi
//sigo:features vfp4,thumb2,fp16
//sigo:float hard

package main

import (
	"sync"
	"time"

	mcu "runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/atsamx51/spi"
	"runtime/arm/cortexm/sam/atsamx51/uart"
	_ "runtime/arm/cortexm/sam/chip/atsame51g19a"
)

var (
	UART = uart.UART5
	SPI  = spi.SPI0
)

func initMCU() {
	defer func() {
		if err := recover(); err != nil {
			UART.WriteString(err.(string) + "\n")
			UART.WriteString("Recovered!\n")
		}
	}()
	defer UART.WriteString("MCU initialized 2\n")
	defer UART.WriteString("MCU initialized 1\n")

	mcu.DefaultClocks()
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

	SPI.Configure(spi.Config{
		DI:             mcu.PA04,
		DO:             mcu.PA07,
		SCK:            mcu.PA05,
		CS:             mcu.PA06,
		BaudHz:         10_000_000,
		CharacterSize:  8,
		DataOrder:      spi.MSB,
		Form:           spi.Frame,
		HardwareSelect: true,
		Mode:           spi.HostMode,
		Phase:          spi.LeadingEdge,
		Polarity:       spi.IdleLow,
		ReceiveEnabled: true,
	})

	panic("testing defers upon panic")
}

func main() {
	initMCU()

	mcu.PB11.SetDirection(1)
	mcu.PB11.Set(true)

	mcu.PB22.SetDirection(0)
	mcu.PB22.SetInterrupt(2, func(pin mcu.Pin) {
		mcu.PB11.Toggle()
	})

	testMap := make(map[string]string)
	testMap["this is"] = "a test\n"

	testMap2 := testMap
	testMap2["this is another"] = "map test\n"

	UART.WriteString(testMap["this is"])
	UART.WriteString(testMap2["this is another"])
	UART.WriteString(testMap2["does not exist"])

	var mutex sync.Mutex
	wait := sync.NewCond(&mutex)

	go func() {
		for {
			time.Sleep(time.Millisecond * 500)
			wait.Signal()
		}
	}()

	go func(mutex sync.Mutex) {
		for {
			wait.Wait()
			mcu.PB11.Toggle()
		}
	}(mutex)

	for {
		//UART.WriteString("test\n")
	}
}
