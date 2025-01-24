package cortexm

import (
	"asm"
	"unsafe"
)

//sigo:export abort runtime.abort
func abort() {
	DisableInterrupts()
	for {
		asm.Inline(`wfi`)
	}
}

//sigo:export currentStack runtime.currentStack
func currentStack() (ptr unsafe.Pointer) {
	asm.Inline(`mrs {{ptr}}, psp`, asm.Out(&ptr))
	return
}

func setCurrentStack(ptr unsafe.Pointer) {
	asm.Inline(`msr psp, {{ptr}}`, asm.In(ptr))
}

//sigo:extern exec runtime.exec
func exec(args unsafe.Pointer, fn unsafe.Pointer)

//sigo:export EnableInterrupts runtime.EnableInterrupts
func EnableInterrupts(state uint32) {
	asm.Inline(`
		msr PRIMASK, {{state}}
		cpsie i
	`, asm.In(state))
}

//sigo:export DisableInterrupts runtime.DisableInterrupts
func DisableInterrupts() (state uint32) {
	asm.Inline(`
		mrs {{state}}, PRIMASK
		cpsid i
	`, asm.Out(&state))
	return
}

//sigo:export InterruptState runtime.InterruptState
func InterruptState() (state uint32) {
	asm.Inline(`
		mrs {{state}}, PRIMASK
	`, asm.Out(&state))
	return
}
