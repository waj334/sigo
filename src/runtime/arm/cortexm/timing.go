package cortexm

import (
	"sync/atomic"

	"runtime/arm/cortexm/support/systemControl"
	"runtime/arm/cortexm/support/systick"
)

var _tickCount uint32
var SysTickFrequency uint32 = 100_000_000

type SysTickSource struct{}

func (s SysTickSource) Now() uint64 {
	// NOTE: The tick count is incremented at a frequency of 1ms
	// Return the tick count in nanoseconds
	return uint64(atomic.LoadUint32(&_tickCount)) * 1_000_000
}

//sigo:export nanotime runtime.nanotime
//sigo:linkage nanotime weak
func nanotime() uint64 {
	state := DisableInterrupts()
	value := uint64(atomic.LoadUint32(&_tickCount)) * 1_000_000
	EnableInterrupts(state)
	return value
}

func initSysTick() {
	// Disable SysTick first
	systick.Systick.Csr.SetEnable(false)

	// Set PendSV to the lowest priority so that context switching does not occur before other interrupts are serviced.
	systemcontrol.SystemControl.Shpr3.SetPri14(5)

	// NOTE: The priority for SysTick should be higher than PendSV, but lower than other critical interrupts.
	systemcontrol.SystemControl.Shpr3.SetPri15(4)

	// TODO: Derive this value from the system clock settings
	UpdateSysTickFrequency(SysTickFrequency)
	systick.Systick.Csr.SetTickint(true)
	systick.Systick.Csr.SetClksource(true)
	systick.Systick.Csr.SetEnable(true)
	for !systick.Systick.Csr.GetEnable() {
	}
}

func UpdateSysTickFrequency(value uint32) {
	SysTickFrequency = value
	systick.Systick.Rvr.SetReload(SysTickFrequency / 1000)
}

//sigo:interrupt SystickHandler SystickHandler
func SystickHandler() {
	// Atomically increment the tick counter
	atomic.AddUint32(&_tickCount, 1)

	// Trigger a PendSV interrupt to run the scheduler
	triggerPendSV()
}

//go:export triggerPendSV runtime.schedulerPause
func triggerPendSV() {
	// Set the PendSV flag
	systemcontrol.SystemControl.Icsr.SetPendsvset(true)
}
