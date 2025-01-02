//go:build atsame5x && !generic && (atsame51j18a || atsame51j19a || atsame51j20a || atsame51n19a || atsame51n20a || atsame54n19a || atsame54n20a || atsame54p19a || atsame54p20a)

package can

var (
	CAN0 = &_can{
		index: 0,
	}

	CAN1 = &_can{
		index: 1,
	}

	instances = [2]*_can{
		CAN0,
		CAN1,
	}
)
