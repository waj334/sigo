
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

    /* Place the stack at the end of RAM so that any stack overflow that occurs
       does not corrupt the heap. */
    .stack (NOLOAD):
    {
        . = ALIGN(4);
        __stack_bottom = .;
        . += __stack_size;
        . = ALIGN(4);
        __stack = .;
    } >RAM
    __stack_top = __stack;

    .data :
    {
        . = ALIGN(4);
        __start_data = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        __end_data = .;
    } >RAM AT>FLASH
    __data_base_addr = LOADADDR(.data);

    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        __start_bss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        __end_bss = .;
    } >RAM

    __heap_start = __end_bss;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM);
    __heap_size = __heap_end - __heap_start;

    /* Exception handling is not needed */
    /DISCARD/ :
    {
        *(.ARM.exidx)
        *(.ARM.exidx.*)
    }
}

__arm32_tls_tcb_offset = 4;
__globals_start = __start_data;
__globals_end = __end_bss;
_estack = __stack;