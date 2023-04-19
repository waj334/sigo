package uart

import "omibyte.io/sigo/src/io"

type UART interface {
	io.ReaderWriter
}
