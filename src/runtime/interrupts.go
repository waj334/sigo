package runtime

//sigo:extern EnableInterrupts runtime.EnableInterrupts
//sigo:extern DisableInterrupts runtime.DisableInterrupts
//sigo:extern InterruptStatus runtime.InterruptStatus
//sigo:extern InInterrupt runtime.InInterrupt

func EnableInterrupts(state uint32)
func DisableInterrupts() uint32
func InterruptStatus() uint32
func InInterrupt() bool

//go:export allowAllocFromInterrupt runtime.allowAllocFromInterrupt
var allowAllocFromInterrupt = false
