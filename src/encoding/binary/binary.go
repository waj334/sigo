package binary

var (
	LittleEndian littleEndian
	BigEndian    bigEndian
)

type ByteOrder interface {
	Uint16(b []byte) uint16
	Uint32(b []byte) uint32
	Uint64(b []byte) uint64
	PutUint16(b []byte, v uint16)
	PutUint32(b []byte, v uint32)
	PutUint64(b []byte, v uint64)
	String() string
}

type AppendByteOrder interface {
	AppendUint16(b []byte, v uint16) []byte
	AppendUint32(b []byte, v uint32) []byte
	AppendUint64(b []byte, v uint64) []byte
	String() string
}

type littleEndian struct{}
type bigEndian struct{}

func (littleEndian) Uint16(b []byte) uint16 {
	return uint16(b[0]) |
		uint16(b[1])<<8
}

func (littleEndian) PutUint16(b []byte, v uint16) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
}

func (e littleEndian) AppendUint16(b []byte, v uint16) []byte {
	return append(b,
		byte(v),
		byte(v>>8))
}

func (littleEndian) Uint32(b []byte) uint32 {
	return uint32(b[0]) |
		uint32(b[1])<<8 |
		uint32(b[2])<<16 |
		uint32(b[3])<<24
}

func (littleEndian) PutUint32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func (e littleEndian) AppendUint32(b []byte, v uint32) []byte {
	return append(b,
		byte(v),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24))
}

func (littleEndian) Uint64(b []byte) uint64 {
	return uint64(b[0]) |
		uint64(b[1])<<8 |
		uint64(b[2])<<16 |
		uint64(b[3])<<24 |
		uint64(b[4])<<32 |
		uint64(b[5])<<40 |
		uint64(b[6])<<48 |
		uint64(b[7])<<56
}

func (littleEndian) PutUint64(b []byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

func (e littleEndian) AppendUint64(b []byte, v uint64) []byte {
	return append(b,
		byte(v),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24),
		byte(v>>32),
		byte(v>>40),
		byte(v>>48),
		byte(v>>56))
}

func (littleEndian) String() string {
	return "LittleEndian"
}

func (bigEndian) Uint16(b []byte) uint16 {
	return uint16(b[1]) |
		uint16(b[0])<<8
}

func (bigEndian) PutUint16(b []byte, v uint16) {
	b[0] = byte(v >> 8)
	b[1] = byte(v)
}

func (e bigEndian) AppendUint16(b []byte, v uint16) []byte {
	return append(b,
		byte(v>>8),
		byte(v))
}

func (bigEndian) Uint32(b []byte) uint32 {
	return uint32(b[3]) |
		uint32(b[2])<<8 |
		uint32(b[1])<<16 |
		uint32(b[0])<<24
}

func (bigEndian) PutUint32(b []byte, v uint32) {
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)
}

func (e bigEndian) AppendUint32(b []byte, v uint32) []byte {
	return append(b,
		byte(v>>24),
		byte(v>>16),
		byte(v>>8),
		byte(v))
}

func (bigEndian) Uint64(b []byte) uint64 {
	return uint64(b[7]) |
		uint64(b[6])<<8 |
		uint64(b[5])<<16 |
		uint64(b[4])<<24 |
		uint64(b[3])<<32 |
		uint64(b[2])<<40 |
		uint64(b[1])<<48 |
		uint64(b[0])<<56
}

func (bigEndian) PutUint64(b []byte, v uint64) {
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
}

func (e bigEndian) AppendUint64(b []byte, v uint64) []byte {
	return append(b,
		byte(v>>56),
		byte(v>>48),
		byte(v>>40),
		byte(v>>32),
		byte(v>>24),
		byte(v>>16),
		byte(v>>8),
		byte(v))
}

func (bigEndian) String() string {
	return "BigEndian"
}
