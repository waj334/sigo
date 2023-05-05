.global _current_stack
.thumb_func
_current_stack:
    mrs r0, msp
    bx lr

.global _abort
.thumb_func
_abort:
    cpsid i
halt:
    wfi
    b halt