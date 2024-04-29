package cortexm

import "unsafe"

//sigo:extern _goroutineStackSize runtime._goroutineStackSize
var _goroutineStackSize uintptr

//sigo:extern _task_start _task_start
var _task_start unsafe.Pointer

type exceptionStack struct {
	R0  uintptr
	R1  uintptr
	R2  uintptr
	R3  uintptr
	R12 uintptr
	LR  uintptr
	PC  uintptr
	PSR uintptr
}

type _task struct {
	stack  unsafe.Pointer
	__func struct {
		ptr  unsafe.Pointer
		args unsafe.Pointer
	}
}

//go:export initTask runtime.initTask
func initTask(taskPtr unsafe.Pointer) {
	task := (*_task)(taskPtr)

	// Calculate the top of the stack past the registers
	// NOTE: The stack grows from the highest address to the lowest address on Cortex-M.
	estack := (*exceptionStack)(unsafe.Add(task.stack, _goroutineStackSize-32))

	// NOTE: The THUMB bit must be set!
	// TODO: This isn't available for Cortex-M0 (excluding Cortex-M0+)
	estack.PSR = 0x0100_0000

	// Set up the call to _task_start
	estack.PC = uintptr(unsafe.Pointer(&_task_start))
	estack.R0 = uintptr(task.__func.args)
	estack.R1 = uintptr(task.__func.ptr)
	estack.R2 = uintptr(taskPtr)

	// Subtract 64 bytes to account for unstacking R4 - R11 during the initial context switch.
	task.stack = unsafe.Add(task.stack, _goroutineStackSize-64)
}

//go:export align runtime.alignStack
func align(n uintptr) uintptr {
	// The stack on Cortex-M is always 8-byte aligned
	return n + (n % 8)
}
