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

      .stack (NOLOAD):
      {
        . = ALIGN(4);
        . += _stack_size; /* Adjust the stack size based on your application's requirements */
        _stack_top = .;
      } >RAM

      .data :
      {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _edata = .;
      } >RAM AT >FLASH

      .bss :
      {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
      } >RAM

    /* Exception handling is not needed */
    /DISCARD/ :
    {
        *(.ARM.exidx)
        *(.ARM.exidx.*)
    }
}

__heap_start = _ebss;
__head_end = ORIGIN(RAM) + LENGTH(RAM);
__globals_start = _sdata;
__globals_end = _ebss;
_estack = _stack_top;

PROVIDE(_stack = _stack_top);