.section .text.Default_Handler
.global  Default_Handler
.type    Default_Handler, %function
Default_Handler:
    wfe
    b    Default_Handler
.size Default_Handler, .-Default_Handler

// Avoid the need for repeated .weak and .set instructions.
.macro IRQ handler
    .weak  \handler
    .set   \handler, Default_Handler
.endm

// Must set the "a" flag on the section:
// https://svnweb.freebsd.org/base/stable/11/sys/arm/arm/locore-v4.S?r1=321049&r2=321048&pathrev=321049
// https://sourceware.org/binutils/docs/as/Section.html#ELF-Version
.section .isr_vector, "a", %progbits
.global  __isr_vector
__isr_vector:
    // Interrupt vector as defined by Cortex-M, starting with the stack top.
    // On reset, SP is initialized with *0x0 and PC is loaded with *0x4, loading
    // _stack_top and Reset_Handler.
    .long __stack
	.long Reset_Handler
	.long NonMaskableInt_Handler
	.long HardFault_Handler
	.long 0
	.long 0
	.long 0
	.long 0
	.long 0
	.long 0
	.long 0
	.long SVCall_Handler
	.long 0
	.long 0
	.long PendSV_Handler
	.long SysTick_Handler
	.long SYSTEM_Handler
	.long WDT_Handler

	IRQ Reset_Handler
    IRQ NonMaskableInt_Handler
    IRQ HardFault_Handler
    IRQ SVCall_Handler
    IRQ PendSV_Handler
    IRQ SysTick_Handler
    IRQ SYSTEM_Handler
    IRQ WDT_Handler

