package runtime

//sigo:extern enableInterrupts _enable_irq
func enableInterrupts(state uint32)

//sigo:extern disableInterrupts _disable_irq
func disableInterrupts() uint32
