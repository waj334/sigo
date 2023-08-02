//go:build generic

package spi

import "io"

type SPI interface {
	io.ReadWriter
	Transact(rx []byte, tx []byte) error
	Select()
	Deselect()
}
