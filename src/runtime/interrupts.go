package runtime

//sigo:extern EnableInterrupts runtime.EnableInterrupts
//sigo:extern DisableInterrupts runtime.DisableInterrupts

func EnableInterrupts(state uint32)
func DisableInterrupts() uint32
