.syntax unified
.thumb
.global runtime.exec
.type runtime.exec, %function
runtime.exec:
    .cfi_startproc
    push {r4-r7, lr}            // Push callee-saved registers onto the stack.
    .cfi_def_cfa_offset 20
    .cfi_offset lr, -4
    .cfi_offset r7, -8
    .cfi_offset r6, -12
    .cfi_offset r5, -16
    .cfi_offset r4, -20

    add r7, sp, #0              // Set up frame pointer for backtracing.
    .cfi_def_cfa_register r7

    blx r1                      // Branch to the function.

    pop {r4-r7, pc}             // Restore the callee-save registers and return.
    .cfi_endproc
