MEMORY
{
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 512K
    RAM (xrw)  : ORIGIN = 0x20000000, LENGTH = 192K
}

_stack_size = 4K;

ENTRY(Reset_Handler)

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
    } >FLASH

    _data_base_addr = LOADADDR(.data);
    .data :
    {
        . = ALIGN(4);
        _start_data = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _end_data = .;
    } >RAM AT>FLASH

    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _start_bss = .;
        *(.sbss*)
        *(.gnu.linkonce.sb.*)
        *(.bss .bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        _end_bss = .;
    } >RAM

    /* Place the stack at the end of RAM so that any stack overflow that occurs
       does not corrupt the heap. */
    .stack (NOLOAD):
    {
        . = ORIGIN(RAM) + LENGTH(RAM) - _stack_size;
        _stack_bottom = .;
        . += _stack_size;
        _stack_top = .;
    } >RAM
    __stack = _stack_top;

    /* Make the rest of memory available for heap storage */
    __heap_start = _end_bss;
    __heap_end = _stack_bottom;
    __heap_size = __heap_end - __heap_start;

    /* Exception handling is not needed */
    /DISCARD/ :
    {
        *(.ARM.exidx)
        *(.ARM.exidx.*)
    }
}

__arm32_tls_tcb_offset = 4;
__globals_start = _start_data;
__globals_end = _end_bss;
_estack = _stack_top;