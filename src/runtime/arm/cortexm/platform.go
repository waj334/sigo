package cortexm

import (
	"asm"
	"asm/register"
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
	asm.Inline(`mrs {ptr}, psp`, asm.Out(&ptr))
	return
}

//sigo:export exec runtime.exec
func exec(args unsafe.Pointer, fn unsafe.Pointer) {
	asm.Inline(`
		mov r0, {args}
		blx {fn}
   `, asm.In(args), asm.In(fn), asm.Clobber(register.R0))
}

//sigo:export EnableInterrupts runtime.EnableInterrupts
func EnableInterrupts(state uint32) {
	asm.Inline(`
		msr PRIMASK, {state}
		cpsie i
	`, asm.In(state))
}

//sigo:export DisableInterrupts runtime.DisableInterrupts
func DisableInterrupts() (state uint32) {
	asm.Inline(`
		mrs {state}, PRIMASK
		cpsid i
	`, asm.Out(&state))
	return
}

//sigo:export InterruptState runtime.InterruptState
func InterruptState() (state uint32) {
	asm.Inline(`
		mrs {state}, PRIMASK
	`, asm.Out(&state))
	return
}
