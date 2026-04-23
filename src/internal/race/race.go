package race

import "unsafe"

const Enabled = false

func Acquire(addr unsafe.Pointer)                       {}
func Release(addr unsafe.Pointer)                       {}
func ReleaseMerge(addr unsafe.Pointer)                  {}
func Disable()                                          {}
func Enable()                                           {}
func Read(addr unsafe.Pointer)                          {}
func ReadPC(addr unsafe.Pointer, callerpc, pc uintptr)  {}
func Write(addr unsafe.Pointer)                         {}
func WritePC(addr unsafe.Pointer, callerpc, pc uintptr) {}
func ReadRange(addr unsafe.Pointer, len int)            {}
func WriteRange(addr unsafe.Pointer, len int)           {}
func Errors() int                                       { return 0 }
