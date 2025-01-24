//go:build generic

package pin

type Pin interface {
	High()
	Low()
	Toggle()

	Set(on bool)
	Get() bool

	SetValue(value int) error
	Value() (int, error)

	SetInterrupt(mode IRQMode, handler func())
	ClearInterrupt()

	SetMode(dir Mode)
	GetMode() Mode

	SetOutputMode(output OutputMode)
	GetOutputMode() OutputMode

	SetSpeedMode(speed SpeedMode)
	GetSpeedMode() SpeedMode

	SetPullMode(mode PullMode)
	GetPullMode() PullMode
}
