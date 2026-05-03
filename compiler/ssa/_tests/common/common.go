// Package common provides shared bring-up code for SSA language tests.
// Each test program calls Setup() from its init function, then writes
// pass/fail markers to UART so the host can observe results.
package common

import (
	"fmt"
	"os"
	"time"

	_ "pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7"
	stm32h7x7 "pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7"
	"pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7/hal"
	"pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7/hal/pin"
	"pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7/hal/timer"
	"pkg.si-go.dev/chip/arm/cortexm/platform/st/stm32h7x7/cm7/hal/uart"
	"pkg.si-go.dev/chip/arm/cortexm/runtime"
)

const TimeScale = uint64(time.Microsecond)

var (
	UART = uart.UART1
	TIM2 = timer.TIM2
)

// Setup configures clocks, UART (for stdout), and TIM2 (so time.Sleep works).
// Call this from each test program's init() function.
func Setup() {
	runtime.SysTickCanWake = false
	hal.ConfigureClocks()

	if err := UART.Configure(uart.Config{
		Enable:          true,
		TX:              pin.PA9,
		RX:              pin.PB7,
		Baud:            115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	}); err != nil {
		panic(err)
	}
	os.Stdout = UART

	if err := TIM2.Configure(timer.Config{Enable: true}); err != nil {
		panic(err)
	}
	stm32h7x7.IrqTim2.SetPriority(1)
}

// Begin announces the start of a test by name. Output is one line.
func Begin(name string) { fmt.Printf("BEGIN %s\n", name) }

// Pass writes a PASS line for a single assertion within a test.
func Pass(label string) { fmt.Printf("PASS %s\n", label) }

// Fail writes a FAIL line with a message. Tests should continue running so the
// host sees as many assertions as possible per build.
func Fail(label string, msg string) { fmt.Printf("FAIL %s: %s\n", label, msg) }

// Done marks the end of a test program. The host watches for this line.
func Done() { fmt.Println("DONE") }

// AssertEq writes PASS if got == want, otherwise FAIL with a description.
func AssertEq(label string, got, want int) {
	if got == want {
		Pass(label)
	} else {
		Fail(label, fmt.Sprintf("got %d, want %d", got, want))
	}
}

// Halt parks the CPU forever. Call after Done() to stop the program.
func Halt() {
	for {
		select {}
	}
}
