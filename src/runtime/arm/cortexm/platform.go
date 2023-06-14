package cortexm

//sigo:extern _enable_irq _enable_irq
func _enable_irq(state uint32)

//sigo:extern _disable_irq _disable_irq
func _disable_irq() uint32

//sigo:extern _irq_state _irq_state
func _irq_state() uint32

func EnableInterrupts(state uint32) {
	_enable_irq(state)
}

func DisableInterrupts() uint32 {
	return _disable_irq()
}

func InterruptState() uint32 {
	return _irq_state()
}
