MEMORY
{
	FLASH (rw) : ORIGIN = 0x0, LENGTH = 512K
	RAM (rw) : ORIGIN = 0x20000000, LENGTH = 512K
}
__stack_size = 4K;
INCLUDE program.ld
