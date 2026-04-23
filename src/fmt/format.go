package fmt

/*
#include <stdio.h>
#include <stdint.h>

static int32_t fmt_int(void *buf, int32_t sz, void *f, int32_t v)     { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_uint(void *buf, int32_t sz, void *f, uint32_t v)   { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_long(void *buf, int32_t sz, void *f, int64_t v)    { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_ulong(void *buf, int32_t sz, void *f, uint64_t v)  { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_double(void *buf, int32_t sz, void *f, double v)   { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_char(void *buf, int32_t sz, void *f, int32_t v)    { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
static int32_t fmt_ptr(void *buf, int32_t sz, void *f, void *v)       { return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v); }
*/
import "C"

import (
	"io"
	"unsafe"
)

type formatter struct {
	w   io.Writer
	n   int
	err error
}

func (f *formatter) write(b []byte) {
	if f.err != nil {
		return
	}
	n, err := f.w.Write(b)
	f.n += n
	if err != nil {
		f.err = err
	}
}

func (f *formatter) writeString(s string) {
	if f.err != nil {
		return
	}
	f.write([]byte(s))
}

func (f *formatter) writeByte(b byte) {
	if f.err != nil {
		return
	}
	var buf [1]byte
	buf[0] = b
	f.write(buf[:])
}

// format parses a Go format string and writes formatted output.
func (f *formatter) format(format string, a []any) {
	argIdx := 0
	i := 0
	for i < len(format) {
		// Find next '%' or end
		start := i
		for i < len(format) && format[i] != '%' {
			i++
		}
		// Write literal segment
		if i > start {
			f.writeString(format[start:i])
		}
		if i >= len(format) {
			break
		}

		// Skip '%'
		i++
		if i >= len(format) {
			f.writeString("%!(NOVERB)")
			break
		}

		// Handle %%
		if format[i] == '%' {
			f.writeByte('%')
			i++
			continue
		}

		// Parse flags
		flagStart := i
		for i < len(format) {
			c := format[i]
			if c == '-' || c == '+' || c == ' ' || c == '0' || c == '#' {
				i++
			} else {
				break
			}
		}
		flags := format[flagStart:i]

		// Parse width
		width := 0
		hasWidth := false
		if i < len(format) && format[i] == '*' {
			// Width from argument
			if argIdx < len(a) {
				if w, ok := a[argIdx].(int); ok {
					width = w
					hasWidth = true
				}
				argIdx++
			}
			i++
		} else {
			for i < len(format) && format[i] >= '0' && format[i] <= '9' {
				width = width*10 + int(format[i]-'0')
				hasWidth = true
				i++
			}
		}

		// Parse precision
		prec := 0
		hasPrec := false
		if i < len(format) && format[i] == '.' {
			i++
			hasPrec = true
			if i < len(format) && format[i] == '*' {
				// Precision from argument
				if argIdx < len(a) {
					if p, ok := a[argIdx].(int); ok {
						prec = p
					}
					argIdx++
				}
				i++
			} else {
				for i < len(format) && format[i] >= '0' && format[i] <= '9' {
					prec = prec*10 + int(format[i]-'0')
					i++
				}
			}
		}

		if i >= len(format) {
			f.writeString("%!(NOVERB)")
			break
		}

		verb := format[i]
		i++

		// Consume argument
		if argIdx < len(a) {
			f.formatArg(verb, flags, width, prec, hasWidth, hasPrec, a[argIdx])
			argIdx++
		} else {
			f.writeString("%!(MISSING)")
		}
	}

	// Extra arguments
	if argIdx < len(a) {
		f.writeString("%!(EXTRA")
		for argIdx < len(a) {
			f.writeByte(' ')
			f.formatArg('v', "", 0, 0, false, false, a[argIdx])
			argIdx++
		}
		f.writeByte(')')
	}
}

// formatArg formats a single argument with the given verb.
func (f *formatter) formatArg(verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	if arg == nil {
		f.writeString("<nil>")
		return
	}

	switch verb {
	case 's', 'w':
		f.formatString(arg)
	case 'd':
		f.formatInt(verb, flags, width, prec, hasWidth, hasPrec, arg)
	case 'x', 'X', 'o':
		f.formatInt(verb, flags, width, prec, hasWidth, hasPrec, arg)
	case 'b':
		f.formatBinary(arg)
	case 'f', 'e', 'E', 'g', 'G':
		f.formatFloat(verb, flags, width, prec, hasWidth, hasPrec, arg)
	case 'c':
		f.formatChar(arg)
	case 'p':
		f.formatPointer(arg)
	case 't':
		f.formatBool(arg)
	case 'v':
		f.formatDefault(flags, width, prec, hasWidth, hasPrec, arg)
	default:
		f.writeString("%!")
		f.writeByte(verb)
		f.writeByte('(')
		f.formatDefault("", 0, 0, false, false, arg)
		f.writeByte(')')
	}
}

// formatString handles %s verb.
func (f *formatter) formatString(arg any) {
	switch v := arg.(type) {
	case string:
		f.writeString(v)
	case error:
		f.writeString(v.Error())
	case Stringer:
		f.writeString(v.String())
	case GoStringer:
		f.writeString(v.GoString())
	default:
		f.writeString("%!s(BADTYPE)")
	}
}

// formatBool handles %t verb.
func (f *formatter) formatBool(arg any) {
	switch v := arg.(type) {
	case bool:
		if v {
			f.writeString("true")
		} else {
			f.writeString("false")
		}
	default:
		f.writeString("%!t(BADTYPE)")
	}
}

// formatChar handles %c verb.
func (f *formatter) formatChar(arg any) {
	var r int
	switch v := arg.(type) {
	case int:
		r = v
	case int8:
		r = int(v)
	case int16:
		r = int(v)
	case int32:
		r = int(v)
	case int64:
		r = int(v)
	case uint:
		r = int(v)
	case uint8:
		r = int(v)
	case uint16:
		r = int(v)
	case uint32:
		r = int(v)
	case uint64:
		r = int(v)
	default:
		f.writeString("%!c(BADTYPE)")
		return
	}

	if r < 128 {
		f.writeByte(byte(r))
	} else {
		// UTF-8 encode
		var buf [4]byte
		n := encodeRune(buf[:], rune(r))
		f.write(buf[:n])
	}
}

// formatPointer handles %p verb.
func (f *formatter) formatPointer(arg any) {
	switch v := arg.(type) {
	case uintptr:
		f.snprintfPtr(unsafe.Pointer(v))
	default:
		f.writeString("%!p(BADTYPE)")
	}
}

// formatBinary handles %b verb (pure Go, since C snprintf may not support %b).
func (f *formatter) formatBinary(arg any) {
	var val uint64
	neg := false
	switch v := arg.(type) {
	case int:
		if v < 0 {
			neg = true
			val = uint64(-v)
		} else {
			val = uint64(v)
		}
	case int8:
		if v < 0 {
			neg = true
			val = uint64(-int64(v))
		} else {
			val = uint64(v)
		}
	case int16:
		if v < 0 {
			neg = true
			val = uint64(-int64(v))
		} else {
			val = uint64(v)
		}
	case int32:
		if v < 0 {
			neg = true
			val = uint64(-int64(v))
		} else {
			val = uint64(v)
		}
	case int64:
		if v < 0 {
			neg = true
			val = uint64(-v)
		} else {
			val = uint64(v)
		}
	case uint:
		val = uint64(v)
	case uint8:
		val = uint64(v)
	case uint16:
		val = uint64(v)
	case uint32:
		val = uint64(v)
	case uint64:
		val = v
	case uintptr:
		val = uint64(v)
	default:
		f.writeString("%!b(BADTYPE)")
		return
	}

	if neg {
		f.writeByte('-')
	}
	if val == 0 {
		f.writeByte('0')
		return
	}

	var buf [64]byte
	i := len(buf)
	for val > 0 {
		i--
		buf[i] = '0' + byte(val&1)
		val >>= 1
	}
	f.write(buf[i:])
}

// formatDefault handles %v verb.
func (f *formatter) formatDefault(flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	switch v := arg.(type) {
	case bool:
		if v {
			f.writeString("true")
		} else {
			f.writeString("false")
		}
	case int:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case int8:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case int16:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case int32:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case int64:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uint:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uint8:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uint16:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uint32:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uint64:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case uintptr:
		f.formatInt('d', flags, width, prec, hasWidth, hasPrec, v)
	case float32:
		f.formatFloat('g', flags, width, prec, hasWidth, hasPrec, v)
	case float64:
		f.formatFloat('g', flags, width, prec, hasWidth, hasPrec, v)
	case string:
		f.writeString(v)
	case error:
		f.writeString(v.Error())
	case Stringer:
		f.writeString(v.String())
	case GoStringer:
		f.writeString(v.GoString())
	default:
		f.writeString("(UNSUPPORTED)")
	}
}

// formatInt handles integer formatting verbs (%d, %x, %X, %o) via snprintf.
func (f *formatter) formatInt(verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	var cfmt [24]byte
	buildCFormat(cfmt[:], verb, flags, width, prec, hasWidth, hasPrec, arg)
	fmtPtr := unsafe.Pointer(&cfmt[0])

	var buf [64]byte
	bufPtr := unsafe.Pointer(&buf[0])
	var written int

	switch v := arg.(type) {
	case int:
		written = int(C.fmt_int(bufPtr, C.int32_t(64), fmtPtr, C.int32_t(v)))
	case int8:
		written = int(C.fmt_int(bufPtr, C.int32_t(64), fmtPtr, C.int32_t(v)))
	case int16:
		written = int(C.fmt_int(bufPtr, C.int32_t(64), fmtPtr, C.int32_t(v)))
	case int32:
		written = int(C.fmt_int(bufPtr, C.int32_t(64), fmtPtr, C.int32_t(v)))
	case int64:
		written = int(C.fmt_long(bufPtr, C.int32_t(64), fmtPtr, C.int64_t(v)))
	case uint:
		written = int(C.fmt_uint(bufPtr, C.int32_t(64), fmtPtr, C.uint32_t(v)))
	case uint8:
		written = int(C.fmt_uint(bufPtr, C.int32_t(64), fmtPtr, C.uint32_t(v)))
	case uint16:
		written = int(C.fmt_uint(bufPtr, C.int32_t(64), fmtPtr, C.uint32_t(v)))
	case uint32:
		written = int(C.fmt_uint(bufPtr, C.int32_t(64), fmtPtr, C.uint32_t(v)))
	case uint64:
		written = int(C.fmt_ulong(bufPtr, C.int32_t(64), fmtPtr, C.uint64_t(v)))
	case uintptr:
		written = int(C.fmt_ulong(bufPtr, C.int32_t(64), fmtPtr, C.uint64_t(v)))
	default:
		f.writeString("%!")
		f.writeByte(verb)
		f.writeString("(BADTYPE)")
		return
	}

	if written > 0 {
		f.write(buf[:written])
	}
}

// formatFloat handles float formatting verbs (%f, %e, %E, %g, %G) via snprintf.
func (f *formatter) formatFloat(verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	var cfmt [24]byte
	buildCFormat(cfmt[:], verb, flags, width, prec, hasWidth, hasPrec, arg)
	fmtPtr := unsafe.Pointer(&cfmt[0])

	var buf [64]byte
	bufPtr := unsafe.Pointer(&buf[0])
	var written int

	switch v := arg.(type) {
	case float32:
		written = int(C.fmt_double(bufPtr, C.int32_t(64), fmtPtr, C.double(float64(v))))
	case float64:
		written = int(C.fmt_double(bufPtr, C.int32_t(64), fmtPtr, C.double(v)))
	default:
		f.writeString("%!")
		f.writeByte(verb)
		f.writeString("(BADTYPE)")
		return
	}

	if written > 0 {
		f.write(buf[:written])
	}
}

// snprintfPtr formats a pointer via snprintf.
func (f *formatter) snprintfPtr(ptr unsafe.Pointer) {
	var cfmt [4]byte
	cfmt[0] = '%'
	cfmt[1] = 'p'
	cfmt[2] = 0

	var buf [24]byte
	written := int(C.fmt_ptr(
		unsafe.Pointer(&buf[0]),
		C.int32_t(24),
		unsafe.Pointer(&cfmt[0]),
		ptr,
	))
	if written > 0 {
		f.write(buf[:written])
	}
}

// defaultFormat formats arguments using default formatting for Print/Fprint/Sprint family.
func (f *formatter) defaultFormat(a []any, addNewline bool) {
	for i, arg := range a {
		if addNewline && i > 0 {
			f.writeByte(' ')
		} else if !addNewline && i > 0 {
			_, prevIsString := a[i-1].(string)
			_, curIsString := arg.(string)
			if !prevIsString && !curIsString {
				f.writeByte(' ')
			}
		}
		f.formatDefault("", 0, 0, false, false, arg)
	}
	if addNewline {
		f.writeByte('\n')
	}
}

// buildCFormat constructs a NUL-terminated C format string in dst.
// Returns the number of bytes written (excluding NUL).
func buildCFormat(dst []byte, verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) int {
	i := 0
	dst[i] = '%'
	i++

	// Copy flags
	for j := 0; j < len(flags); j++ {
		dst[i] = flags[j]
		i++
	}

	// Width
	if hasWidth {
		i += writeIntToBytes(dst[i:], width)
	}

	// Precision
	if hasPrec {
		dst[i] = '.'
		i++
		i += writeIntToBytes(dst[i:], prec)
	}

	// Length modifier for 64-bit types
	switch arg.(type) {
	case int64:
		dst[i] = 'l'
		i++
		dst[i] = 'l'
		i++
	case uint64:
		dst[i] = 'l'
		i++
		dst[i] = 'l'
		i++
	case uintptr:
		dst[i] = 'l'
		i++
		dst[i] = 'l'
		i++
	default:
		// No length modifier needed for 32-bit and smaller types.
	}

	// For unsigned verbs with signed int types, use unsigned verb directly
	// (the caller already casts to the right C type)
	dst[i] = verb
	i++

	// NUL terminate
	dst[i] = 0

	return i
}

// writeIntToBytes writes a non-negative integer as decimal digits into dst.
// Returns the number of bytes written.
func writeIntToBytes(dst []byte, val int) int {
	if val == 0 {
		dst[0] = '0'
		return 1
	}
	if val < 0 {
		val = -val
	}

	var tmp [10]byte
	i := len(tmp)
	for val > 0 {
		i--
		tmp[i] = byte('0' + val%10)
		val /= 10
	}

	n := len(tmp) - i
	copy(dst, tmp[i:])
	return n
}

// encodeRune encodes a rune as UTF-8 into buf, returning the number of bytes written.
func encodeRune(buf []byte, r rune) int {
	u := uint32(r)
	if u < 0x80 {
		buf[0] = byte(u)
		return 1
	} else if u < 0x800 {
		buf[0] = byte(0xC0 | (u >> 6))
		buf[1] = byte(0x80 | (u & 0x3F))
		return 2
	} else if u < 0x10000 {
		buf[0] = byte(0xE0 | (u >> 12))
		buf[1] = byte(0x80 | ((u >> 6) & 0x3F))
		buf[2] = byte(0x80 | (u & 0x3F))
		return 3
	} else {
		buf[0] = byte(0xF0 | (u >> 18))
		buf[1] = byte(0x80 | ((u >> 12) & 0x3F))
		buf[2] = byte(0x80 | ((u >> 6) & 0x3F))
		buf[3] = byte(0x80 | (u & 0x3F))
		return 4
	}
}
