package peripheral

type PinDirection int
type PinIRQMode int
type PinPullMode int

type PinInterruptHandler func(Pin)

type Pin interface {
	High()
	Low()
	Toggle()

	Set(on bool)
	Get() bool

	SetInterrupt(mode PinIRQMode, handler func(Pin))
	ClearInterrupt()

	SetDirection(dir PinDirection)
	GetDirection() PinDirection

	SetPullMode(mode PinPullMode)
	GetPullMode() PinPullMode
}
