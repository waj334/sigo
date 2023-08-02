//go:build generic

package uart

import "io"

type UART interface {
	io.ReadWriter
	io.StringWriter
}
