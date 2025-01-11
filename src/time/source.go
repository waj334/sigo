package time

type Source interface {
	Now() (nsec uint64)
}
