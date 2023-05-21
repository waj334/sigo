package cortexm

//go:linkname EnableInterrupts _enable_irq
func EnableInterrupts()

//go:linkname DisableInterrupts _disable_irq
func DisableInterrupts()
