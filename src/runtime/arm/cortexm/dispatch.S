.global _dispatch
.type _dispatch, %function
_dispatch:
    push {r4-r9, lr}            // Save callee-saved registers and return address
    mov r10, sp                 // Store the original stack pointer in r10 for cleanup later

    mov r4, r0                  // Move the function pointer into r4
    mov r5, r1                  // Move the arguments into r5

    ldr r6, [r5]                // Load the argument count from r5
    adds r5, r5, #4             // Advance to the first argument

    mov r8, #0                  // Initialize the counter to zero in r8
.iterate:
    cmp r8, r6                  // Compare the count to the argument count
    beq .invoke                 // Invoke if the arguments have been exhausted

    ldr r7, [r5]                // Load the size of the parameter from the address stored in r5 into r7
    adds r5, r5, #4             // Advance to the value of the first argument

    cmp r8, #0
    beq .reg0                   // Load the next parameter into r0
    cmp r8, #1
    beq .reg1                   // Load the next parameter into r1
    cmp r8, #2
    beq .reg2                   // Load the next parameter into r2
    cmp r8, #3
    beq .reg3                   // Load the next parameter into r3
    b .stacksave                // Push the next parameter onto the stack

.reg0:
    ldr r0, [r5]                // Load the parameter into r0
    b .advance

.reg1:
    ldr r1, [r5]                // Load the parameter into r1
    b .advance

.reg2:
    ldr r2, [r5]                // Load the parameter into r2
    b .advance

.reg3:
    ldr r3, [r5]                // Load the parameter into r3
    b .advance

.stacksave:
    ldr r9, [r5]                // Load the value of the parameter at the address stored in r5
    push {r9}                   // Push the parameter onto the stack
    b .advance

.advance:
    adds r8, r8, #1             // Advance the counter
    adds r5, r7                 // Advance to the size of the next parameter
    b .iterate                  // Prepare the next argument

.invoke:
    blx r4                      // Call the function
    mov sp, r10                 // Restore the original stack pointer to preserve any value returned by the callee
    pop {r4-r9, pc}             // Restore the callee-saved registers and return

.size _dispatch, .-_dispatch