//go:build generic

package i2c

import "io"

type I2C interface {
	io.ReadWriter
	SetAddress(addr uint16)
	SetClockFrequency(clockSpeedHz uint32) bool
	GetClockFrequency() uint32
	WriteAddress(addr uint16, b []byte) (n int, err error)
	ReadAddress(addr uint16, b []byte) (n int, err error)
}
