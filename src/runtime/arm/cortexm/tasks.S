.global _task_start
.type _task_start, %function
_task_start:
    push {r2}                       // Save the task pointer on the stack
    blx r1                          // Start the goroutine. The parameters are already on R0
    pop {r0}                        // Pop the task pointer into R0 as the parameter to removeTask
    bl runtime.removeTask           // Remove this task from the scheduler
    bl runtime.schedulerPause       // Run another task for the remainder of this SysTick
.loop:                              // ---------------------------
    nop                             // Loop until PendSV is eventually serviced
    b .loop                         // ---------------------------
.size _task_start, .-_task_start

.global runtime.currentTask
.global runtime.lastTask
.global PendSV_Handler
.type PendSV_Handler, %function
PendSV_Handler:
    // Choose the next task
    bl runtime.runScheduler
    cmp r0, #0                      // Check if runScheduler returned false
    beq _return                     // No context switch should be performed

    // Load the stack pointer of the current task
    ldr r0, =runtime.currentTask    // Load the address of the current task pointer into R0
    ldr r0, [r0]                    // Load the current task into R0
    ldr r1, =runtime.lastTask       // Load the last task pointer into R1
    ldr r1, [r1]                    // Load the value of the last task pointer into R1
    cmp r1, r0                      // Compare the last task to the current task
    beq _return                     // Do nothing if the task has not changed

    ldr r0, [r0]                    // Load the stack pointer into R0
    cmp r1, #0                      // Check if the value of the pointer to the last task is 0
    beq _context_switch             // Skip saving the context of the last task if there was no last task

    // Save context of old task
    mrs r1, psp;                    // Get the current PSP
#ifdef __thumb2__
    stmdb r1!, {r4-r11}             // Save r4-r11 on the last task's stack
#else
    subs r1, r1, #4*8               // Update the stack pointer

    str r4, [r1, #0]                // Store r4 on the stack
    str r5, [r1, #4]                // Store r5 on the stack
    str r6, [r1, #4*2]              // Store r6 on the stack
    str r7, [r1, #4*3]              // Store r7 on the stack

    mov r2, r8                      // Copy r8 to r2
    str r2, [r1, #4*4]              // Store r8 on the stack
    mov r2, r9                      // Copy r9 to r2
    str r2, [r1, #4*5]              // Store r9 on the stack
    mov r2, r10                     // Copy r10 to r2
    str r2, [r1, #4*6]              // Store r10 on the stack
    mov r2, r11                     // Copy r11 to r2
    str r2, [r1, #4*7]              // Store r11 on the stack

#endif
    // Save the stack pointer of the last task
    ldr r2, =runtime.lastTask       // Load the last task pointer into R1
    ldr r2, [r2]                    // Load the value of the last task pointer into R1
    str r1, [r2]                    // Store updated stack pointer in the task's context
_context_switch:
    // Load context of new task
#ifdef __thumb2__
    ldmia r0!, {r4-r11}             // Load r4-r11 and lr from the new task's stack
#else
    ldr r4, [r0, #4*0]              // Load r4 from the new task's stack
    ldr r5, [r0, #4*1]              // Load r5 from the new task's stack
    ldr r6, [r0, #4*2]              // Load r6 from the new task's stack
    ldr r7, [r0, #4*3]              // Load r7 from the new task's stack
    ldr r3, [r0, #4*4]              // Load r8 from the new task's stack into r3
    mov r8, r3                      // Move r3 to r8
    ldr r3, [r0, #4*5]              // Load r9 from the new task's stack into r3
    mov r9, r3                      // Move r3 to r9
    ldr r3, [r0, #4*6]              // Load r10 from the new task's stack into r3
    mov r10, r3                     // Move r3 to r10
    ldr r3, [r0, #4*7]              // Load r11 from the new task's stack into r3
    mov r11, r3                     // Move r3 to r11
    adds r0, r0, #4*8
#endif
    msr psp, r0;                    // Update PSP
_return:
#ifdef __thumb2__
    ldr lr, =0xFFFFFFFD             // Set EXC_RETURN value for return to Thread mode with PSP
#else
    ldr r3, =0xFFFFFFFD
    mov lr, r3
#endif
    bx lr                           // Use EXC_RETURN to set the next PC
.size PendSV_Handler, .-PendSV_Handler