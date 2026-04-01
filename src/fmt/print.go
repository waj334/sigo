package fmt

import (
	"io"
	"os"
)

// byteBuffer is a simple growable buffer implementing io.Writer.
type byteBuffer struct {
	buf []byte
}

func (b *byteBuffer) Write(p []byte) (int, error) {
	b.buf = append(b.buf, p...)
	return len(p), nil
}

func (b *byteBuffer) WriteString(s string) (int, error) {
	b.buf = append(b.buf, s...)
	return len(s), nil
}

// Fprintf formats according to a format string and writes to w.
func Fprintf(w io.Writer, format string, a ...any) (int, error) {
	f := formatter{w: w}
	f.format(format, a)
	return f.n, f.err
}

// Printf formats according to a format string and writes to standard output.
func Printf(format string, a ...any) (int, error) {
	return Fprintf(os.Stdout, format, a...)
}

// Sprintf formats according to a format string and returns the resulting string.
func Sprintf(format string, a ...any) string {
	var buf byteBuffer
	f := formatter{w: &buf}
	f.format(format, a)
	return string(buf.buf)
}

// Fprint formats using the default formats for its operands and writes to w.
// Spaces are added between operands when neither is a string.
func Fprint(w io.Writer, a ...any) (int, error) {
	f := formatter{w: w}
	f.defaultFormat(a, false)
	return f.n, f.err
}

// Fprintln formats using the default formats for its operands and writes to w.
// Spaces are always added between operands and a newline is appended.
func Fprintln(w io.Writer, a ...any) (int, error) {
	f := formatter{w: w}
	f.defaultFormat(a, true)
	return f.n, f.err
}

// Print formats using the default formats for its operands and writes to standard output.
// Spaces are added between operands when neither is a string.
func Print(a ...any) (int, error) {
	return Fprint(os.Stdout, a...)
}

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
func Println(a ...any) (int, error) {
	return Fprintln(os.Stdout, a...)
}

// Sprint formats using the default formats for its operands and returns the resulting string.
// Spaces are added between operands when neither is a string.
func Sprint(a ...any) string {
	var buf byteBuffer
	f := formatter{w: &buf}
	f.defaultFormat(a, false)
	return string(buf.buf)
}

// Sprintln formats using the default formats for its operands and returns the resulting string.
// Spaces are always added between operands and a newline is appended.
func Sprintln(a ...any) string {
	var buf byteBuffer
	f := formatter{w: &buf}
	f.defaultFormat(a, true)
	return string(buf.buf)
}
