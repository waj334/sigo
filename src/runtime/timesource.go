package runtime

type TimeSource interface {
	Now() (nsec uint64)
}

type SysTickSource struct{}

func (s *SysTickSource) Now() uint64 {
	// NOTE: The tick count is incremented at a frequency of 1ms
	// Return the tick count in nanoseconds
	return uint64(_currentTick()) * 1_000_000
}
