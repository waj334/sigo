package net

import (
	"runtime"
	"time"
)

// sys_arch_protect / sys_arch_unprotect — lwIP critical section port (NO_SYS=1).
// sys_prot_t is `int` in C (arch/cc.h); mapped to int32 here.

//go:export sysArchProtect sys_arch_protect
func sysArchProtect() int32 {
	return int32(runtime.DisableInterrupts())
}

//go:export sysArchUnprotect sys_arch_unprotect
func sysArchUnprotect(pval int32) {
	runtime.EnableInterrupts(uint32(pval))
}

// sys_now returns milliseconds elapsed; used by lwIP timeout management.

//go:export sysNow sys_now
func sysNow() uint32 {
	return uint32(time.Now().UnixMilli())
}
