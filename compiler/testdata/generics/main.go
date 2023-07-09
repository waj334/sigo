//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi
//sigo:features vfp4,thumb2,fp16
//sigo:float hard

package main

import (
	"time"

	mcu "runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/atsamx51/uart"
	_ "runtime/arm/cortexm/sam/chip/atsame51g19a"
)

type Printer interface {
	Print()
}

type PrinterA struct{}

func (p PrinterA) Print() {
	uart.UART5.WriteString("PrinterA\n")
}

type PrinterB struct{}

func (p PrinterB) Print() {
	uart.UART5.WriteString("PrinterB\n")
}

func Print[T Printer](p T) {
	p.Print()
}

func initMCU() {
	defer uart.UART5.WriteString("MCU initialized 1\n")
	mcu.DefaultClocks()
	uart.UART5.Configure(uart.Config{
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

	blinkChan := make(chan struct{})
	blinkChan2 := make(chan struct{})

	go func() {
		for {
			time.Sleep(time.Millisecond * 500)
			blinkChan <- struct{}{}
			time.Sleep(time.Millisecond * 500)
			blinkChan2 <- struct{}{}
		}
	}()

	go func(blinkChan, blinkChan2 chan struct{}) {
		for {
			select {
			case <-blinkChan:
				mcu.PB11.Toggle()
				Print(PrinterA{})
			case <-blinkChan2:
				Print(PrinterB{})
				mcu.PB11.Toggle()
				time.Sleep(time.Millisecond * 100)
				mcu.PB11.Toggle()
				time.Sleep(time.Millisecond * 100)
				mcu.PB11.Toggle()
			}
		}
	}(blinkChan, blinkChan2)

	select {}
}
