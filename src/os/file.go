package os

import "io"

type iostream interface {
	io.Reader
	io.Writer
	io.StringWriter
}

var (
	Stdin  iostream
	Stdout iostream
	Stderr iostream
)
