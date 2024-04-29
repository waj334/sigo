package cortexm

import (
	"sync/atomic"
)

var _tickCount uint32
var SYSTICK_FREQUENCY uint32
var NPRIORITY_BITS uint8 = 8

//go:export currentTick runtime.currentTick
func currentTick() uint32 {
	return atomic.LoadUint32(&_tickCount)
}

//sigo:extern runScheduler runtime.runScheduler
func runScheduler() bool

func initSysTick() {
	// Disable SysTick first
	SYST.CSR.SetENABLE(false)

	// Set PendSV to the lowest priority so that context switching does not occur before other interrupts are serviced.
	SCS.SHPR3.SetPRI_14(0xFF)

	// NOTE: The priority for SysTick should be higher than PendSV, but lower than other critical interrupts.
	SCS.SHPR3.SetPRI_15(4)

	// TODO: Derive this value from the system clock settings
	SYST.RVR.SetRELOAD(SYSTICK_FREQUENCY / 1000)
	SYST.CSR.SetTICKINT(true)
	SYST.CSR.SetCLKSOURCE(true)
	SYST.CSR.SetENABLE(true)
	for !SYST.CSR.GetENABLE() {
	}
}

//sigo:interrupt _SysTick_Handler SysTick_Handler
func _SysTick_Handler() {
	// Atomically increment the tick counter
	atomic.AddUint32(&_tickCount, 1)

	// Trigger a PendSV interrupt to run the scheduler
	triggerPendSV()
}

//go:export triggerPendSV runtime.schedulerPause
func triggerPendSV() {
	// Set the PendSV flag
	SCS.ICSR.SetPENDSVSET(true)
}
