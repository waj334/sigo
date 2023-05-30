package runtime

//go:linkname enableInterrupts _enable_irq
func enableInterrupts()

//go:linkname disableInterrupts _disable_irq
func disableInterrupts()
