package peripheral

import "io"

type SPI interface {
	io.ReadWriter
	Transact(b []byte) []byte
	Select()
	Deselect()
}
