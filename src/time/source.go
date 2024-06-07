package time

type TimeSource interface {
	Now() (nsec uint64)
}
