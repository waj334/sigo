//go:build atsamx5x

package main

import (
	"peripheral/pin"
	"peripheral/rtc"
	"peripheral/uart"
	"time"

	mcu "runtime/arm/cortexm/sam/samx51"
)

//sigo:export nanotime runtime.nanotime
func nanotime() uint64 {
	// The runtime and time package will use the RTC as its time source.
	return RTC.Now()
}

var (
	RTC  = rtc.RTC
	UART = uart.UART5
	LED  = pin.PB11
)

func init() {
	rtc.SOURCE_CLK_FREQUENCY = 32_768 // Hz

	// Initialize the clock system.
	mcu.DefaultClocks()

	// Configure UART.
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
		OnCompare0:   blink,
	}); err != nil {
		UART.WriteString(err.Error() + "\n")
		panic(err)
	}

	// Set up the LED.
	LED.SetDirection(pin.Output)
	LED.Set(true)
}

func main() {
	// Block forever.
	select {}
}

func blink() {
	// Get the current compare values.
	cmp := RTC.CompareValue()

	// Set the first value to now + 500ms.
	cmp[0] = RTC.Value() + RTC.Ticks(time.Millisecond*500)

	// Apply the new compare value.
	RTC.SetCompareValue(cmp)

	// Toggle the LED.
	LED.Toggle()
}
