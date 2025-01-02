//go:build atsame5x && !generic && (atsame51g18a || atsame51g19a)

package can

var (
	CAN0 = &_can{
		index: 0,
	}

	instances = [1]*_can{
		CAN0,
	}
)
