//go:build arm

package register

import (
	"asm"
)

const (
	// L In Thumb2 mode, low 32-bit GPR registers (r0-r7). In Thumb1 mode, A low 32-bit GPR register (r0-r7). In ARM mode, same as r.
	L asm.RegisterClass = "l"

	// H In Thumb2 mode, a high 32-bit GPR register (r8-r15). In Thumb1 mode, A high 32-bit GPR register (r0-r7). In ARM mode, invalid.
	H asm.RegisterClass = "h"

	// W A 32, 64, or 128-bit floating-point/SIMD register in the ranges s0-s31, d0-d31, or q0-q15, respectively.
	W asm.RegisterClass = "w"

	// T A 32, 64, or 128-bit floating-point/SIMD register in the ranges s0-s31, d0-d15, or q0-q7, respectively.
	T asm.RegisterClass = "t"

	// X A 32, 64, or 128-bit floating-point/SIMD register in the ranges s0-s15, d0-d7, or q0-q3, respectively.
	X asm.RegisterClass = "x"

	R0  asm.Register = "r0"
	R1  asm.Register = "r1"
	R2  asm.Register = "r2"
	R3  asm.Register = "r3"
	R4  asm.Register = "r4"
	R5  asm.Register = "r5"
	R6  asm.Register = "r6"
	R7  asm.Register = "r7"
	R8  asm.Register = "r8"
	R9  asm.Register = "r9"
	R10 asm.Register = "r10"
	R11 asm.Register = "r11"
	R12 asm.Register = "r12"
	R13 asm.Register = "r13"
	R14 asm.Register = "r14"
	R15 asm.Register = "r15"

	RFP = R9
	SL  = R10
	FP  = R11
	IP  = R12
	SP  = R13
	LR  = R14
	PC  = R15
)
