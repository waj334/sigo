//go:build generic

package can

import "io"

type CAN interface {
	io.ReaderWriter
	SendFrame(frame Frame) error
	ReceiveFrame() (Frame, error)
}
