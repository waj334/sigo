// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

    .syntax unified
    .thumb
    .text

// func memequal(a, b unsafe.Pointer, size uintptr) bool
    .global runtime_memequal
    .type   runtime_memequal, %function
runtime_memequal:
    ldr     r0, [sp, #0]        // a
    ldr     r2, [sp, #4]        // b
    cmp     r2, r0              // b - a
    beq     .Lmemequal_eq
    ldr     r1, [sp, #8]        // size
    cmp     r1, #0
    beq     .Lmemequal_eq
    add     r7, sp, #12         // r7 = &ret
    b       .Lmemeqbody
.Lmemequal_eq:
    mov     r0, #1
    strb    r0, [sp, #12]       // ret = true
    bx      lr
    .size   runtime_memequal, .-runtime_memequal

// func memequal_varlen(a, b unsafe.Pointer) bool
// R7 on entry is the closure pointer (Go ARM ABI); size is at closure+4.
    .global runtime_memequal_varlen
    .type   runtime_memequal_varlen, %function
runtime_memequal_varlen:
    ldr     r0, [sp, #0]        // a
    ldr     r2, [sp, #4]        // b
    cmp     r2, r0              // b - a
    beq     .Lmemequal_varlen_eq
    ldr     r1, [r7, #4]        // size from closure (compiler stores it at offset 4)
    cmp     r1, #0
    beq     .Lmemequal_varlen_eq
    add     r7, sp, #8          // r7 = &ret  (reassign closure ptr to return slot)
    b       .Lmemeqbody
.Lmemequal_varlen_eq:
    mov     r0, #1
    strb    r0, [sp, #8]        // ret = true
    bx      lr
    .size   runtime_memequal_varlen, .-runtime_memequal_varlen

// Input:
//   r0 = data of a
//   r1 = length
//   r2 = data of b
//   r7 = pointer to return byte
// Clobbers: r4, r5, r6
    .type   .Lmemeqbody, %function
.Lmemeqbody:
    cmp     r1, #1
    beq     .Lone               // 1-byte fast path
    cmp     r1, #4
    add     r1, r1, r0          // r1 = a + len  (end pointer)
    blt     .Lbyte_loop         // len < 4
    and     r6, r0, #3
    cmp     r6, #0
    bne     .Lbyte_loop         // a unaligned
    and     r6, r2, #3
    cmp     r6, #0
    bne     .Lbyte_loop         // b unaligned
    bic     r6, r1, #3          // r6 = r1 & ~3  (word-aligned end)
.Lchunk4_loop:
    ldr     r4, [r0], #4        // MOVW.P 4(R0), R4
    ldr     r5, [r2], #4        // MOVW.P 4(R2), R5
    cmp     r5, r4
    bne     .Lnotequal
    cmp     r6, r0
    bne     .Lchunk4_loop
    cmp     r1, r0
    beq     .Lequal             // reached end
.Lbyte_loop:
    ldrb    r4, [r0], #1        // MOVBU.P 1(R0), R4
    ldrb    r5, [r2], #1        // MOVBU.P 1(R2), R5
    cmp     r5, r4
    bne     .Lnotequal
    cmp     r1, r0
    bne     .Lbyte_loop
.Lequal:
    mov     r0, #1
    strb    r0, [r7]
    bx      lr
.Lone:
    ldrb    r4, [r0]            // MOVBU (R0), R4  (no post-increment)
    ldrb    r5, [r2]            // MOVBU (R2), R5
    cmp     r5, r4
    beq     .Lequal
.Lnotequal:
    mov     r0, #0
    strb    r0, [r7]
    bx      lr
    .size   .Lmemeqbody, .-.Lmemeqbody