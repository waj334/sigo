// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

    .syntax unified
    .thumb
    .text

// func IndexByte(b []byte, c byte) int
    .global IndexByte
    .type   IndexByte, %function
IndexByte:
    ldr     r0, [sp, #0]        // b_base
    ldr     r1, [sp, #4]        // b_len
    ldrb    r2, [sp, #12]       // c
    add     r5, sp, #16         // r5 = &ret
    b       .Lindexbytebody
    .size   IndexByte, .-IndexByte

// func IndexByteString(s string, c byte) int
    .global IndexByteString
    .type   IndexByteString, %function
IndexByteString:
    ldr     r0, [sp, #0]        // s_base
    ldr     r1, [sp, #4]        // s_len
    ldrb    r2, [sp, #8]        // c
    add     r5, sp, #12         // r5 = &ret
    b       .Lindexbytebody
    .size   IndexByteString, .-IndexByteString

// Input:
//   r0 = data pointer
//   r1 = data length
//   r2 = byte to find
//   r5 = address to store result
// Clobbers: r3, r4
    .type   .Lindexbytebody, %function
.Lindexbytebody:
    mov     r4, r0              // r4 = base (save for index calculation)
    add     r1, r1, r0          // r1 = data + len  (end pointer)
.Lloop:
    cmp     r1, r0              // end - current
    beq     .Lnotfound
    ldrb    r3, [r0], #1        // MOVBU.P 1(R0), R3
    cmp     r3, r2
    bne     .Lloop
    sub     r0, r0, #1          // post-increment overshot by 1
    sub     r0, r0, r4          // index = ptr - base
    str     r0, [r5]
    bx      lr
.Lnotfound:
    mvn     r0, #0              // r0 = -1
    str     r0, [r5]
    bx      lr
    .size   .Lindexbytebody, .-.Lindexbytebody