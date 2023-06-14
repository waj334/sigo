//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	"sync"
	"time"

	_ "runtime/arm/cortexm/sam/atsame51g19a"
	mcu "runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/atsamx51/uart"
)

var (
	UART = uart.UART5
)

func initMCU() {
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
}

func main() {
	initMCU()

	mcu.PB11.SetDirection(1)
	mcu.PB11.Set(true)

	mcu.PB22.SetDirection(0)
	mcu.PB22.SetInterrupt(2, func(pin mcu.Pin) {
		mcu.PB11.Toggle()
	})

	var mutex sync.Mutex

	go func() {
		for {
			time.Sleep(time.Second * 5)
			mutex.Lock()
			time.Sleep(time.Second * 5)
			mutex.Unlock()
		}
	}()

	go func(mutex sync.Mutex) {
		for {
			time.Sleep(time.Second)
			mutex.Lock()
			mcu.PB11.Toggle()
			mutex.Unlock()
		}
	}(mutex)

	for {
		UART.WriteString("test\n")
	}
}
