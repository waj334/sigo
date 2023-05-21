package peripheral

import "io"

type UART interface {
	io.ReadWriter
	io.StringWriter
}
