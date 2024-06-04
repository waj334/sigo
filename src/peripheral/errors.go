package peripheral

const (
	ErrInvalidPinout Error = -1
	ErrInvalidConfig Error = -2
	ErrInvalidBuffer Error = -3
)

type Error int

func (e Error) Error() string {
	switch e {
	case 0:
		return "no error"
	case -1:
		return "invalid pinout"
	case -2:
		return "invalid configuration"
	case -3:
		return "invalid buffer"
	default:
		return "unknown error"
	}
}
