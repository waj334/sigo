//go:build generic

package pin

type Pin interface {
	High()
	Low()
	Toggle()

	Set(on bool)
	Get() bool

	SetInterrupt(mode IRQMode, handler func(Pin))
	ClearInterrupt()

	SetDirection(dir Direction)
	GetDirection() Direction

	SetPullMode(mode PullMode)
	GetPullMode() PullMode
}
