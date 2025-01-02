//go:build atsamd21

package main

import (
	"time"

	"peripheral/pin"

	"runtime/arm/cortexm/sam/atsamd21"
)

var (
	LED = pin.PA17
)

func main() {
	atsamd21.DefaultClocks()

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
