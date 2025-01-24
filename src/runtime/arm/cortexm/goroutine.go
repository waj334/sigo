package cortexm

import (
	"nonstandard"
	"unsafe"
)

//sigo:extern _goroutineStackSize runtime._goroutineStackSize
var _goroutineStackSize uintptr

const basicFrameSize = unsafe.Sizeof(basicFrame{})
const basicGoroutineContextSize = unsafe.Sizeof(basicGoroutineContext{})

//go:extern removeGoroutine  runtime.removeGoroutine
func removeGoroutine(unsafe.Pointer)

type basicFrame struct {
	R0  uintptr
	R1  uintptr
	R2  uintptr
	R3  uintptr
	R12 uintptr
	LR  uintptr
	PC  uintptr
	PSR uintptr
}

type stackFrame struct {
	basicFrame
	extendedFrame
}

type basicGoroutineContext struct {
	R4  uintptr
	R5  uintptr
	R6  uintptr
	R7  uintptr
	R8  uintptr
	R9  uintptr
	R10 uintptr
	R11 uintptr
	LR  uintptr
}
type goroutineContext struct {
	basicGoroutineContext
	extendedGoroutineContext
}

type _goroutine struct {
	stack unsafe.Pointer
	fn    struct {
		ptr  unsafe.Pointer
		args unsafe.Pointer
	}
}

//go:export initGoroutine runtime.initGoroutine
func initGoroutine(gptr unsafe.Pointer) {
	g := (*_goroutine)(gptr)

	// Calculate the top of the stack past the registers
	// NOTE: The stack grows from the highest address to the lowest address on Cortex-M.
	estack := (*stackFrame)(unsafe.Add(g.stack, _goroutineStackSize-basicFrameSize))

	// Set up the call to startGoroutine.
	estack.PSR = defaultPsrValue

	estack.PC = uintptr(nonstandard.PointerOf(startGoroutine))
	estack.R0 = uintptr(gptr)

	// estack.PC = uintptr(unsafe.Pointer(&fnPtrStartGoroutine))
	// estack.R0 = uintptr(g.fn.args)
	// estack.R1 = uintptr(g.fn.ptr)
	// estack.R2 = uintptr(gptr)

	// Subtract 2 stack stackFrame lengths to account for unstacking R4 - R11 during the initial context switch.
	g.stack = unsafe.Add(unsafe.Pointer(estack), -int32(basicGoroutineContextSize))
	ctx := (*goroutineContext)(g.stack)
	ctx.R4 = 0x0101_0101
	ctx.R5 = 0x0202_0202
	ctx.R6 = 0x0303_0303
	ctx.R7 = 0x0404_0404
	ctx.R8 = 0x0505_0505
	ctx.R9 = 0x0606_0606
	ctx.R10 = 0x0707_0707
	ctx.R11 = 0x0808_0808
	ctx.LR = 0xFFFF_FFFD
}

//go:export align runtime.alignStack
func align(n uintptr) uintptr {
	// The stack on Cortex-M is always 8-byte aligned
	return n + (n % 8)
}

//sigo:export startGoroutine
//sigo:attribute startGoroutine noreturn
func startGoroutine(g *_goroutine) {
	// Start the goroutine.
	exec(g.fn.args, g.fn.ptr)

	// Remove the goroutine from the scheduler.
	removeGoroutine(unsafe.Pointer(g))

	// Busy loop until another task can be run.
	for {
		// Trigger a context switch.
		triggerPendSV()
	}
}
