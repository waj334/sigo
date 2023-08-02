//go:build samd21

package main

import (
	"peripheral/pin"
	mcu "runtime/arm/cortexm/sam/samd21"
	"time"
)

var (
	LED = pin.PA17
)

func initMCU() {
	mcu.DefaultClocks()
}

func main() {
	initMCU()

	LED.SetDirection(pin.Output)
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
