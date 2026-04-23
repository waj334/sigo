// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

    .syntax unified
    .thumb
    .text

// func Compare(a, b []byte) int
    .global Compare
    .type   Compare, %function
Compare:
    ldr     r2, [sp, #0]        // a_base
    ldr     r0, [sp, #4]        // a_len
    ldr     r3, [sp, #12]       // b_base
    ldr     r1, [sp, #16]       // b_len
    add     r7, sp, #28         // r7 = &retval
    b       .Lcmpbody
    .size   Compare, .-Compare

// func cmpstring(a, b string) int
    .global runtime_cmpstring
    .type   runtime_cmpstring, %function
runtime_cmpstring:
    ldr     r2, [sp, #0]        // a_base
    ldr     r0, [sp, #4]        // a_len
    ldr     r3, [sp, #8]        // b_base
    ldr     r1, [sp, #12]       // b_len
    add     r7, sp, #20         // r7 = &retval
    b       .Lcmpbody
    .size   runtime_cmpstring, .-runtime_cmpstring

// On entry:
//   r0 = len(a), r1 = len(b)
//   r2 = a ptr,  r3 = b ptr
//   r7 = pointer to return slot
// Clobbers: r4, r5, r6, r8
    .type   .Lcmpbody, %function
.Lcmpbody:
    cmp     r3, r2
    beq     .Lsamebytes
    cmp     r1, r0
    mov     r6, r0
    it      lt
    movlt   r6, r1              // r6 = min(len(a), len(b))
    cmp     r6, #0
    beq     .Lsamebytes
    cmp     r6, #4
    add     r6, r6, r2          // r6 = a + min_len  (end of compare range)
    blt     .Lbyte_loop
    and     r8, r2, #3
    cmp     r8, #0
    bne     .Lbyte_loop         // a unaligned
.Laligned_a:
    and     r8, r3, #3
    cmp     r8, #0
    bne     .Lbyte_loop         // b unaligned
    bic     r8, r6, #3          // r8 = r6 & ~3  (word-aligned chunk end)
.Lchunk4_loop:
    ldr     r4, [r2], #4
    ldr     r5, [r3], #4
    cmp     r5, r4
    bne     .Lcmp
    cmp     r8, r2
    bne     .Lchunk4_loop
    cmp     r6, r2
    beq     .Lsamebytes
.Lbyte_loop:
    ldrb    r4, [r2], #1
    ldrb    r5, [r3], #1
    cmp     r5, r4
    bne     .Lret
    cmp     r6, r2
    bne     .Lbyte_loop

.Lsamebytes:
    // return sign of len(b) - len(a)
    cmp     r1, r0
    beq     .Lsamebytes_eq
    bgt     .Lsamebytes_neg     // len(b) > len(a): a < b → -1
    mov     r0, #1              // len(b) < len(a): a > b → +1
    str     r0, [r7]
    bx      lr
.Lsamebytes_neg:
    mvn     r0, #0              // -1
    str     r0, [r7]
    bx      lr
.Lsamebytes_eq:
    mov     r0, #0
    str     r0, [r7]
    bx      lr

.Lret:
    // bytes differed; flags still set from cmp r5, r4  (r5 - r4)
    bgt     .Lret_neg           // r5 > r4: b's byte > a's byte: a < b → -1
    mov     r0, #1              // r5 < r4: b's byte < a's byte: a > b → +1
    str     r0, [r7]
    bx      lr
.Lret_neg:
    mvn     r0, #0              // -1
    str     r0, [r7]
    bx      lr

.Lcmp:
    sub     r2, r2, #4          // undo post-increment
    sub     r3, r3, #4
    b       .Lbyte_loop
    .size   .Lcmpbody, .-.Lcmpbody