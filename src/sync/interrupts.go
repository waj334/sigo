package sync

//sigo:extern enableInterrupts runtime.EnableInterrupts
//sigo:extern disableInterrupts runtime.DisableInterrupts

func enableInterrupts(state uint32)
func disableInterrupts() uint32
