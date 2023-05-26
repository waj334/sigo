package time

var (
	source Source
)

func SetSource(src Source) {
	source = src
}

type Source interface {
	Now() (nsec uintptr)
}

type nullSource struct{}

func (s *nullSource) Now() uintptr {
	return 0
}
