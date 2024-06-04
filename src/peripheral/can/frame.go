package can

type DataLengthCode uint8

const (
	DLC8 DataLengthCode = iota
	DLC12
	DLC16
	DLC20
	DLC24
	DLC32
	DLC48
	DLC64

	headerLength = 8
)

func dataLengthInBytes(code DataLengthCode) int {
	switch code {
	case DLC8:
		return 8
	case DLC12:
		return 12
	case DLC16:
		return 16
	case DLC20:
		return 20
	case DLC24:
		return 24
	case DLC32:
		return 32
	case DLC48:
		return 48
	case DLC64:
		return 64
	default:
		panic("invalid data length code")
	}
}

func ByteLengthToDataLengthCode(length int) DataLengthCode {
	switch length {
	case 8:
		return DLC8
	case 12:
		return DLC12
	case 16:
		return DLC16
	case 20:
		return DLC20
	case 24:
		return DLC24
	case 32:
		return DLC32
	case 48:
		return DLC48
	case 64:
		return DLC64
	default:
		// Nonstandard. The callee will have to deal with this.
		return DataLengthCode(length)
	}
}

type Frame struct {
	ID             uint32
	Data           []byte
	DataLengthCode DataLengthCode
	Extended       bool
	Nil            bool
	FD             bool

	// Store the buffer so the underlying data is not garbage collected.
	buf []byte
}
