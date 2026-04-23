// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

    .syntax unified
    .thumb
    .text

// func Count(b []byte, c byte) int
    .global Count
    .type   Count, %function
Count:
    ldr     r0, [sp, #0]        // b_base
    ldr     r1, [sp, #4]        // b_len
    ldrb    r2, [sp, #12]       // c
    add     r7, sp, #16         // r7 = &ret
    b       .Lcountbytebody
    .size   Count, .-Count

// func CountString(s string, c byte) int
    .global CountString
    .type   CountString, %function
CountString:
    ldr     r0, [sp, #0]        // s_base
    ldr     r1, [sp, #4]        // s_len
    ldrb    r2, [sp, #8]        // c
    add     r7, sp, #12         // r7 = &ret
    b       .Lcountbytebody
    .size   CountString, .-CountString

// Input:
//   r0 = data pointer
//   r1 = data length
//   r2 = byte to find
//   r7 = address to store result
// Clobbers: r4, r8
    .type   .Lcountbytebody, %function
.Lcountbytebody:
    mov     r8, #0              // r8 = count
    cmp     r1, #0
    beq     .Ldone
    add     r1, r1, r0          // r1 = data + len  (end pointer)
.Lbyte_loop:
    ldrb    r4, [r0], #1
    cmp     r2, r4
    it      eq
    addeq   r8, r8, #1
    cmp     r1, r0
    bne     .Lbyte_loop
.Ldone:
    str     r8, [r7]
    bx      lr
    .size   .Lcountbytebody, .-.Lcountbytebody