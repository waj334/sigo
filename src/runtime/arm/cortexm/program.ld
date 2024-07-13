
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
        . = ALIGN(8);

        /* Required for global ctors. */
        PROVIDE_HIDDEN ( __init_array_start = . );
        KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP (*(.init_array .ctors))
        PROVIDE_HIDDEN ( __init_array_end = . );

    } >FLASH

    /* Place the stack at the end of RAM so that any stack overflow that occurs
       does not corrupt the heap. */
    .stack (NOLOAD):
    {
        __stack_bottom = .;
        . += __stack_size;
        . = ALIGN(8);
        __stack_top = .;
    } >RAM

    .data :
    {
        __start_data = .;
        *(.data)
        *(.data.*)
        . = ALIGN(8);
        __end_data = .;
    } >RAM AT>FLASH
    __data_base_addr = LOADADDR(.data);

    .bss (NOLOAD) :
    {
        __start_bss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(8);
        __end_bss = .;
    } >RAM

    __heap_start = __end_bss;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM);
    __heap_size = __heap_end - __heap_start;

    /* Define the range that the garbage collector
       should scan for heap references. */
    __gc_scan_start = __start_data;
    __gc_scan_end   = __end_bss;

    /* Exception handling is not needed. */
    /DISCARD/ :
    {
        *(.note .note.*)
        *(.eh_frame .eh_frame.*)
        *(.ARM.extab* .gnu.linkonce.armextab.*)
        *(.ARM.exidx*)
    }
}

__globals_start = __start_data;
__globals_end = __end_bss;
_estack = __stack_top;
__stack = __stack_top;