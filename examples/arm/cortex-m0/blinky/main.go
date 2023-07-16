//sigo:architecture arm
//sigo:cpu cortex-m0
//sigo:triple armv6m-none-eabi
//sigo:features

package main

import (
	"time"

	mcu "runtime/arm/cortexm/sam/atsamd21"
	_ "runtime/arm/cortexm/sam/chip/atsamd21g18a"
)

var (
	LED = mcu.PA17
)

func initMCU() {
	mcu.DefaultClocks()
}

func main() {
	initMCU()

	LED.SetDirection(mcu.Output)
	LED.Set(true)

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
				LED.Toggle()
			case <-blinkChan2:
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
