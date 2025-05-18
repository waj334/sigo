.global runtime.currentGoroutine
.global runtime.lastGoroutine
.global PendsvHandler
.type PendsvHandler, %function
PendsvHandler:
    push {lr}                           // Temporarily store LR on the stack since the branch below will NOT restore LR.
    bl runtime.runScheduler             // Choose the next goroutine
#ifdef __thumb2__
    pop {lr}                            // Restore the previous value of LR
#else
    pop {r1}
    mov lr, r0
#endif
    cmp r0, #0                          // Check if runScheduler returned false
    beq _restore_and_return             // No context switch should be performed

    ldr r0, =runtime.currentGoroutine   // Load the address of the current goroutine pointer into R0
    ldr r0, [r0]                        // Load the current goroutine into R0
    ldr r1, =runtime.lastGoroutine      // Load the last goroutine pointer into R1
    ldr r1, [r1]                        // Load the value of the last goroutine pointer into R1
    cmp r1, r0                          // Compare the last goroutine to the current goroutine
    beq _restore_and_return             // Do nothing if the goroutine has not changed

    ldr r0, [r0]                        // Load the stack pointer into R0
    cmp r1, #0                          // Check if the value of the pointer to the last goroutine is 0
    beq _context_switch                 // Skip saving the context of the last goroutine if there was no last goroutine

    mrs r1, psp;                        // Get the current PSP
#ifdef __thumb2__
    tst lr, #0x10                       // Is the goroutine using the FPU context?
    it eq
    vstmdbeq r1!, {s16-s31}             // Save the high VFP registers first if the FPU was active
    stmdb r1!, {r4-r11, lr}             // Save [R4-R11] and LR on the last goroutine's stack
#else
    // NOTE: No supported armv6-m targets have a FPU, so no FPU context is considered!

    subs r1, r1, #4*8                   // Update the stack pointer to allocate space for [R4-R11]

    str r4, [r1, #0]                    // Store R4 on the stack
    str r5, [r1, #4]                    // Store R5 on the stack
    str r6, [r1, #4*2]                  // Store R6 on the stack
    str r7, [r1, #4*3]                  // Store R7 on the stack

    mov r2, r8                          // Copy R8 into R2
    str r2, [r1, #4*4]                  // Store R8 on the stack
    mov r2, r9                          // Copy R9 into R2
    str r2, [r1, #4*5]                  // Store R9 on the stack
    mov r2, r10                         // Copy R10 into R2
    str r2, [r1, #4*6]                  // Store R10 on the stack
    mov r2, r11                         // Copy R11 into R2
    str r2, [r1, #4*7]                  // Store R11 on the stack
#endif
    ldr r2, =runtime.lastGoroutine      // Load the last goroutine pointer into R2
    ldr r2, [r2]                        // Load the address of the last goroutine pointer into R2
    str r1, [r2]                        // Store updated stack pointer in the goroutine's context

_context_switch:
#ifdef __thumb2__
    ldmia r0!, {r4-r12}                 // Load [R4-R12] (where R12 holds the previous value of LR) from the new goroutine's stack
    tst r12, #0x10                      // Is the goroutine using the FPU context?
    itte eq
    vldmiaeq r0!, {s16-s31}             // Load the high VFP registers
    ldreq lr, =0xFFFFFFED               // Set EXC_RETURN value for return to Thread mode with PSP and FPU enabled
    ldrne lr, =0xFFFFFFFD               // Set EXC_RETURN value for return to Thread mode with PSP
#else
    ldr r4, [r0, #4*0]                  // Load R4 from the new goroutine's stack
    ldr r5, [r0, #4*1]                  // Load R5 from the new goroutine's stack
    ldr r6, [r0, #4*2]                  // Load R6 from the new goroutine's stack
    ldr r7, [r0, #4*3]                  // Load R7 from the new goroutine's stack
    ldr r3, [r0, #4*4]                  // Load R8 from the new goroutine's stack into R3
    mov r8, r3                          // Move R3 to R8
    ldr r3, [r0, #4*5]                  // Load R9 from the new goroutine's stack into R3
    mov r9, r3                          // Move R3 to R9
    ldr r3, [r0, #4*6]                  // Load R10 from the new goroutine's stack into R3
    mov r10, r3                         // Move R3 to R10
    ldr r3, [r0, #4*7]                  // Load R11 from the new goroutine's stack into R3
    mov r11, r3                         // Move R3 to R11
    adds r0, r0, #4*8
#endif
    msr psp, r0;                        // Update the program stack pointer before returning
    bx lr                               // Continue execution at the next goroutine

_restore_and_return:
    bx lr                               // Continue executing the last goroutine
.size PendsvHandler, .-PendsvHandler
.align 4
