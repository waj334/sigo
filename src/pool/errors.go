package pool

import "errors"

// Errors returned by Pool constructors.
var (
	ErrInvalidAlignment = errors.New("pool: alignment must be a non-zero power of two")
	ErrInvalidSlotSize  = errors.New("pool: slot size must be non-zero")
	ErrBufferTooSmall   = errors.New("pool: backing buffer too small after alignment")
)
