package peripheral

import "errors"

var (
	ErrInvalidPinout = errors.New("invalid pinout")
	ErrInvalidConfig = errors.New("invalid configuration")
)
