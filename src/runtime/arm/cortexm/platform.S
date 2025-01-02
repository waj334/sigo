.global runtime.exec
.type runtime.exec, %function
runtime.exec:
	mov r4, lr
    blx r1
	mov lr, r4
	bx lr
.size runtime.exec, .-runtime.exec

.global runtime.currentStack
.type runtime.currentStack, %function
runtime.currentStack:
    mrs r0, psp
    bx lr

.global runtime.abort
.type runtime.abort, %function
runtime.abort:
    cpsid i
halt:
    wfi
    b halt

.global _enable_irq
.type _enable_irq, %function
_enable_irq:
    msr PRIMASK, r0
    cpsie i
    bx lr

.global _disable_irq
.type _disable_irq, %function
_disable_irq:
    mrs r0, PRIMASK
    cpsid i
    bx lr

.global _irq_state
.type _irq_state, %function
_irq_state:
    mrs r0, PRIMASK
    bx lr

.section .text.HardfaultHandler
.global HardfaultHandler
.type HardfaultHandler. %function
HardfaultHandler:
#ifdef __thumb2__
    tst lr, #4
    ite eq
    mrseq r0, msp
    mrsne r0, psp
    bl _hardfault
#else
    mov r1, lr
    lsrs r1, #2
    bcc use_msp
    mrs r0, psp
    bl _hardfault

use_msp:
    mrs r0, msp
    bl _hardfault
#endif
