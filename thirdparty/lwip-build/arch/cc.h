#ifndef LWIP_ARCH_CC_H
#define LWIP_ARCH_CC_H

#ifdef LITTLE_ENDIAN
#undef LITTLE_ENDIAN
#endif
#ifdef BIG_ENDIAN
#undef BIG_ENDIAN
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* Type definitions using stdint */
typedef uint8_t   u8_t;
typedef int8_t    s8_t;
typedef uint16_t  u16_t;
typedef int16_t   s16_t;
typedef uint32_t  u32_t;
typedef int32_t   s32_t;
typedef uintptr_t mem_ptr_t;

/* printf formatters */
#define U16_F "u"
#define S16_F "d"
#define X16_F "x"
#define U32_F "lu"
#define S32_F "ld"
#define X32_F "lx"

/* Compiler hints for packing */
#define PACK_STRUCT_FIELD(x) x
#define PACK_STRUCT_STRUCT   __attribute__((packed))
#define PACK_STRUCT_BEGIN
#define PACK_STRUCT_END

/* Interrupt protection type for NO_SYS mode */
typedef int sys_prot_t;

/* Diagnostics */
#define LWIP_PLATFORM_DIAG(x) do { printf x; } while(0)
#define LWIP_PLATFORM_ASSERT(x) do { printf("Assertion \"%s\" failed at %s:%d\n", \
    x, __FILE__, __LINE__); abort(); } while(0)

#endif /* LWIP_ARCH_CC_H */
