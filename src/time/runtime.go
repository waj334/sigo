package time

import "unsafe"

//sigo:extern runtime.gopark
func gopark(unsafe.Pointer)

//sigo:extern runtime.goparkRestore
func goparkRestore(unsafe.Pointer, uint32)

//sigo:extern runtime.goresume
func goresume(unsafe.Pointer)

//sigo:extern runtime.goready
func goready(ptr unsafe.Pointer) bool

//sigo:extern runtime.getgPtr
func getg() unsafe.Pointer

//sigo:extern runtime.DisableInterrupts
func DisableInterrupts() uint32

//sigo:extern runtime.EnableInterrupts
func EnableInterrupts(uint32)

//sigo:extern runtime.nanotime
func nanotime() uint64
