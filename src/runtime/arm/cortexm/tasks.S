.global _task_start
.type _task_start, %function
_task_start:
    push {r2}                       // Save the task pointer on the stack
    blx r1                          // Start the goroutine. The parameters are already on R0
    pop {r0}                        // Pop the task pointer into R0 as the parameter to removeTask
    bl runtime.removeTask           // Remove this task from the scheduler
    bl _triggerPendSV               // Run another task for the remainder of this SysTick
.loop:                              // ---------------------------
    nop                             // Loop until PendSV is eventually serviced
    b .loop                         // ---------------------------
.size _task_start, .-_task_start

.global runtime._currentTask
.global PendSV_Handler
.type PendSV_Handler, %function
PendSV_Handler:
    // Load the stack pointer of the current task
    ldr r0, =runtime._currentTask   // Load the address of the current task pointer into R0
    ldr r0, [r0]                    // Load the current task into R0
    ldr r0, [r0]                    // Load the stack pointer into R0

    // Save context of old task
    mrs r1, psp;                    // Get the current PSP
    subs r1, r1, #4*9               // Make room for r4-r11
#ifdef __thumb2__
    stmia r1!, {r4-r11, lr}         // Save r4-r11 on the stack
#else
    str r4, [r1], #4                // Store r4 on the stack
    str r5, [r1], #4                // Store r5 on the stack
    str r6, [r1], #4                // Store r6 on the stack
    str r7, [r1], #4                // Store r7 on the stack
    str r8, [r1], #4                // Store r8 on the stack
    str r9, [r1], #4                // Store r9 on the stack
    str r10, [r1], #4               // Store r10 on the stack
    str r11, [r1], #4               // Store r11 on the stack
    str lr, [r1]                    // Store lr on the stack
#endif
    msr psp, r1                     // Update PSP

    // Load context of new task
    subs r0, r0, #4*9               // r4-r11 are stored at a lower address on the stack
#ifdef __thumb2__
    ldmia r0!, {r4-r11, lr}         // Load r4-r11 and lr from the new task's stack
#else
    ldr r4, [r0], #4                // Load r4 from the new task's stack
    ldr r5, [r0], #4                // Load r5 from the new task's stack
    ldr r6, [r0], #4                // Load r6 from the new task's stack
    ldr r7, [r0], #4                // Load r7 from the new task's stack
    ldr r8, [r0], #4                // Load r8 from the new task's stack
    ldr r9, [r0], #4                // Load r9 from the new task's stack
    ldr r10, [r0], #4               // Load r10 from the new task's stack
    ldr r11, [r0], #4               // Load r11 from the new task's stack
    ldr lr, [r0]                    // Load lr from the new task's stack
#endif
    msr psp, r0;                    // Update PSP
    ldr lr, =0xFFFFFFFD             // Set EXC_RETURN value for return to Thread mode with PSP
    bx lr                           // Use EXC_RETURN to set the next PC
.size PendSV_Handler, .-PendSV_Handler