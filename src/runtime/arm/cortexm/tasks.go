package cortexm

import "unsafe"

//sigo:extern _goroutineStackSize runtime._goroutineStackSize
var _goroutineStackSize *uintptr

//sigo:extern _task_start _task_start
var _task_start unsafe.Pointer

type exceptionStack struct {
	regs registers
	R0   uintptr
	R1   uintptr
	R2   uintptr
	R3   uintptr
	R12  uintptr
	LR   uintptr
	PC   uintptr
	PSR  uintptr
}

type registers struct {
	R4  uintptr
	R5  uintptr
	R6  uintptr
	R7  uintptr
	R8  uintptr
	R9  uintptr
	R10 uintptr
	R11 uintptr
}

type _task struct {
	stack     unsafe.Pointer
	goroutine *struct {
		fn   unsafe.Pointer
		args unsafe.Pointer
	}
}

//go:export initTask runtime.initTask
func initTask(taskPtr unsafe.Pointer) {
	task := (*_task)(taskPtr)
	estack := (*exceptionStack)(unsafe.Add(task.stack, *_goroutineStackSize-unsafe.Sizeof(exceptionStack{})))

	// NOTE: The THUMB bit must be set!
	// TODO: This isn't available for Cortex-M0
	estack.PSR = 0x01000000

	// Set up the call to _task_start
	estack.PC = uintptr(_task_start)
	estack.R0 = uintptr(task.goroutine.args)
	estack.R1 = uintptr(task.goroutine.fn)
	estack.R2 = uintptr(taskPtr)
	task.stack = unsafe.Pointer(estack)
}

//go:export align runtime.alignStack
func align(n uintptr) uintptr {
	// The stack on Cortex-M is always 8-byte aligned
	return n + (n % 8)
}
