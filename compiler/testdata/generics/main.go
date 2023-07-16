//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi
//sigo:features vfp4,thumb2,fp16
//sigo:float hard

package main

import (
	_ "reflect"
	"strconv"
	"time"

	mcu "runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/atsamx51/uart"
	_ "runtime/arm/cortexm/sam/chip/atsame51g19a"
)

var (
	UART   = uart.UART5
	LED    = mcu.PB11
	BUTTON = mcu.PB22
)

type Printer interface {
	Print()
}

type PrinterA struct{}

func (p PrinterA) Print() {
	UART.WriteString("PrinterA\n")
}

type PrinterB struct{}

func (p PrinterB) Print() {
	UART.WriteString("PrinterB\n")
}

func Print[T Printer](p T) {
	p.Print()
}

func initMCU() {
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
}

func main() {
	initMCU()

	LED.SetDirection(mcu.Output)
	LED.Set(true)

	BUTTON.SetDirection(mcu.Input)
	BUTTON.SetInterrupt(mcu.FallingEdge, func(mcu.Pin) {
		LED.Toggle()
	})

	blinkChan := make(chan struct{})
	blinkChan2 := make(chan struct{})

	testMap := make(map[string]string)
	testMap["1"] = "a"
	testMap["2"] = "b"
	testMap["3"] = "c"
	testMap["4"] = "d"

	for k, v := range testMap {
		UART.WriteString("testMap[" + k + "]=" + v + "\n")
	}

	i := 0

	go func() {
		for {
			time.Sleep(time.Millisecond * 500)
			blinkChan <- struct{}{}
			time.Sleep(time.Millisecond * 500)
			blinkChan2 <- struct{}{}
			UART.WriteString(strconv.Itoa(i) + "\n")
			i++
		}
	}()

	go func(blinkChan, blinkChan2 chan struct{}) {
		for {
			select {
			case <-blinkChan:
				LED.Toggle()
				Print(PrinterA{})
			case <-blinkChan2:
				Print(PrinterB{})
				LED.Toggle()
				time.Sleep(time.Millisecond * 100)
				LED.Toggle()
				time.Sleep(time.Millisecond * 100)
				LED.Toggle()
			}
		}
	}(blinkChan, blinkChan2)

	select {}
}
