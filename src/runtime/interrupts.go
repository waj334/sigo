package runtime

//sigo:extern enableInterrupts _enable_irq
func enableInterrupts(state uint32)

//sigo:extern disableInterrupts _disable_irq
func disableInterrupts() uint32

func EnableInterrupts(state uint32) {
	enableInterrupts(state)
}

func DisableInterrupts() uint32 {
	return disableInterrupts()
}
