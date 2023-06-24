package atsamx51

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
	"sync/atomic"
)

var _tickCount uint32

//go:export currentTick runtime.currentTick
func currentTick() uint32 {
	return atomic.LoadUint32(&_tickCount)
}

//sigo:extern runScheduler runtime.runScheduler
func runScheduler() bool

//go:export initSysTick initSysTick
func initSysTick() {
	// Set up priorities
	chip.SystemControl.SHPR3.SetPRI_14(0) // Set PendSV to the highest priority
	chip.SystemControl.SHPR3.SetPRI_15(1) // Set SysTick below PendSV

	// TODO: Derive this value from the system clock settings
	chip.SysTick.RVR.SetRELOAD(uint32(GCLK0_FREQUENCY) / 1000)
	chip.SysTick.CSR.SetTICKINT(chip.SysTickCSRTICKINTSelectVALUE_1)
	chip.SysTick.CSR.SetCLKSOURCE(chip.SysTickCSRCLKSOURCESelectVALUE_1)
	chip.SysTick.CSR.SetENABLE(chip.SysTickCSRENABLESelectVALUE_1)
	for chip.SysTick.CSR.GetENABLE() == 0 {
	}
}

//sigo:interrupt _SysTick_Handler SysTick_Handler
func _SysTick_Handler() {
	state := cortexm.DisableInterrupts()
	// Atomically increment the tick counter
	atomic.AddUint32(&_tickCount, 1)

	// Trigger a PendSV interrupt to run the scheduler
	triggerPendSV()
	cortexm.EnableInterrupts(state)
}

//go:export triggerPendSV runtime.schedulerPause
func triggerPendSV() {
	// Set the PendSV flag
	chip.SystemControl.ICSR.SetPENDSVSET(chip.SystemControlICSRPENDSVSETSelectVALUE_1)
}
