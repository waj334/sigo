package peripheral

type PinInterruptHandler func(Pin)

type Pin interface {
	High()
	Low()
	Toggle()

	Set(on bool)
	Get() bool

	SetInterrupt(mode int, handler func(Pin))
	ClearInterrupt()

	SetDirection(dir int)
	GetDirection() int

	SetPullMode(mode int)
	GetPullMode() int
}
