.global _current_stack
.type _current_stack, %function
_current_stack:
    mrs r0, psp
    bx lr
.size _current_stack, .-_current_stack

.global _abort
.type _abort, %function
_abort:
    cpsid i
halt:
    wfi
    b halt
.size _abort, .-_abort

.global _enable_irq
.type _enable_irq, %function
_enable_irq:
    cpsie i
    bx lr
.size _enable_irq, .-_enable_irq

.global _disable_irq
.type _disable_irq, %function
_disable_irq:
    cpsid i
    bx lr
.size _disable_irq, .-_disable_irq

.section .text.HardFault_Handler
.global HardFault_Handler
.type HardFault_Handler. %function
HardFault_Handler:
    mov r0, sp
    movs r3, #0
    ldr r3, [r3]
    bl _hardfault
.size HardFault_Handler, .-HardFault_Handler
