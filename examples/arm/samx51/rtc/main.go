//go:build samx51

package main

import (
	"peripheral/pin"
	"peripheral/rtc"
	"peripheral/uart"
	mcu "runtime/arm/cortexm/sam/samx51"
	"time"
)

var (
	RTC  = rtc.RTC
	UART = uart.UART5
	LED  = pin.PB11
)

func init() {
	rtc.SOURCE_CLK_FREQUENCY = 32_768

	// Use the realtime clock as the time source.
	time.SetSource(RTC)
}

func main() {
	// Initialize the clock system
	mcu.DefaultClocks()

	// Configure UART
	UART.Configure(uart.Config{
		TXD:             pin.PB02,
		RXD:             pin.PB03,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	// Configure the realtime clock.
	if err := RTC.Configure(rtc.Config{
		Prescaler:    rtc.DIV1,
		ClearOnMatch: false,
		Value:        0,
		Compare:      [2]uint32{0, 0},
	}); err != nil {
		UART.WriteString(err.Error() + "\n")
		panic(err)
	}

	// Set up the LED
	LED.SetDirection(pin.Output)
	LED.Set(true)

	go func(LED pin.Pin) {
		// Blink the LED
		for {
			time.Sleep(time.Millisecond * 500)
			LED.Toggle()
		}
	}(LED)

	select {}
}
