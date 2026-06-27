package fmt

/*
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// All scalar values cross the Go/C boundary by pointer. This avoids depending
// on the caller and Clang agreeing about the ABI locations of 64-bit and
// floating-point arguments. memcpy also avoids alignment and aliasing issues.
static int32_t fmt_int(void *buf, int32_t sz, void *f, const void *vp) {
    int32_t v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}

static int32_t fmt_uint(void *buf, int32_t sz, void *f, const void *vp) {
    uint32_t v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}

static int32_t fmt_long(void *buf, int32_t sz, void *f, const void *vp) {
    int64_t v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}

static int32_t fmt_ulong(void *buf, int32_t sz, void *f, const void *vp) {
    uint64_t v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}

static int32_t fmt_double(void *buf, int32_t sz, void *f, const void *vp) {
    double v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}

static int32_t fmt_ptr(void *buf, int32_t sz, void *f, const void *vp) {
    void *v;
    memcpy(&v, vp, sizeof(v));
    return (int32_t)snprintf((char*)buf, (size_t)sz, (const char*)f, v);
}
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
	case 'q':
		f.formatQuoted(arg)
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

// formatQuoted handles %q verb.
func (f *formatter) formatQuoted(arg any) {
	switch v := arg.(type) {
	case string:
		f.writeString(`"`)
		f.writeString(v)
		f.writeString(`"`)
	case error:
		f.writeString(`"`)
		f.writeString(v.Error())
		f.writeString(`"`)
	case Stringer:
		f.writeString(`"`)
		f.writeString(v.String())
		f.writeString(`"`)
	case GoStringer:
		f.writeString(`"`)
		f.writeString(v.GoString())
		f.writeString(`"`)
	default:
		f.writeString("%!q(BADTYPE)")
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

type cFormatKind uint8

const (
	cFormatInt32 cFormatKind = iota
	cFormatUint32
	cFormatInt64
	cFormatUint64
	cFormatDouble
	cFormatPointer
)

const initialCFormatBufferSize = 64

func cFormatIsSigned(kind cFormatKind) bool {
	return kind == cFormatInt32 || kind == cFormatInt64
}

func cFormatIsUnsigned(kind cFormatKind) bool {
	return kind == cFormatUint32 || kind == cFormatUint64
}

func cFormatIs64Bit(kind cFormatKind) bool {
	return kind == cFormatInt64 || kind == cFormatUint64
}

// callCFormat invokes a fixed-signature C wrapper. Values are always passed by
// pointer so that no integer-pair or VFP argument ABI is involved at this
// boundary.
func callCFormat(kind cFormatKind, buf unsafe.Pointer, size int, format, value unsafe.Pointer) int {
	switch kind {
	case cFormatInt32:
		return int(C.fmt_int(buf, C.int32_t(size), format, value))
	case cFormatUint32:
		return int(C.fmt_uint(buf, C.int32_t(size), format, value))
	case cFormatInt64:
		return int(C.fmt_long(buf, C.int32_t(size), format, value))
	case cFormatUint64:
		return int(C.fmt_ulong(buf, C.int32_t(size), format, value))
	case cFormatDouble:
		return int(C.fmt_double(buf, C.int32_t(size), format, value))
	case cFormatPointer:
		return int(C.fmt_ptr(buf, C.int32_t(size), format, value))
	default:
		return -1
	}
}

func (f *formatter) writeCFormatError(verb byte) {
	f.writeString("%!")
	f.writeByte(verb)
	f.writeString("(FORMATERR)")
}

// writeCFormatted performs snprintf with a stack buffer first, then retries
// with exactly enough storage when snprintf reports a larger required size.
// snprintf returns the number of bytes that would have been written, excluding
// the trailing NUL, so that value must never be used directly as a slice bound
// unless it is smaller than the supplied buffer.
func (f *formatter) writeCFormatted(verb byte, kind cFormatKind, format, value unsafe.Pointer) {
	var initial [initialCFormatBufferSize]byte
	written := callCFormat(
		kind,
		unsafe.Pointer(&initial[0]),
		len(initial),
		format,
		value,
	)
	if written < 0 {
		f.writeCFormatError(verb)
		return
	}
	if written < len(initial) {
		f.write(initial[:written])
		return
	}

	maxInt := int(^uint(0) >> 1)
	for {
		// written+1 is required for snprintf's trailing NUL.
		if written >= maxInt {
			f.writeCFormatError(verb)
			return
		}

		buf := make([]byte, written+1)
		n := callCFormat(
			kind,
			unsafe.Pointer(&buf[0]),
			len(buf),
			format,
			value,
		)
		if n < 0 {
			f.writeCFormatError(verb)
			return
		}
		if n < len(buf) {
			f.write(buf[:n])
			return
		}

		// The formatted size should normally be stable. Retry if the C library
		// reports a larger size on the second call rather than slicing past buf.
		written = n
	}
}

// formatInt handles integer formatting verbs (%d, %x, %X, %o) via snprintf.
func (f *formatter) formatInt(verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	var kind cFormatKind
	var value unsafe.Pointer

	var i32 int32
	var u32 uint32
	var i64 int64
	var u64 uint64

	switch v := arg.(type) {
	case int:
		if unsafe.Sizeof(v) == 8 {
			i64 = int64(v)
			kind = cFormatInt64
			value = unsafe.Pointer(&i64)
		} else {
			i32 = int32(v)
			kind = cFormatInt32
			value = unsafe.Pointer(&i32)
		}
	case int8:
		i32 = int32(v)
		kind = cFormatInt32
		value = unsafe.Pointer(&i32)
	case int16:
		i32 = int32(v)
		kind = cFormatInt32
		value = unsafe.Pointer(&i32)
	case int32:
		i32 = v
		kind = cFormatInt32
		value = unsafe.Pointer(&i32)
	case int64:
		i64 = v
		kind = cFormatInt64
		value = unsafe.Pointer(&i64)
	case uint:
		if unsafe.Sizeof(v) == 8 {
			u64 = uint64(v)
			kind = cFormatUint64
			value = unsafe.Pointer(&u64)
		} else {
			u32 = uint32(v)
			kind = cFormatUint32
			value = unsafe.Pointer(&u32)
		}
	case uint8:
		u32 = uint32(v)
		kind = cFormatUint32
		value = unsafe.Pointer(&u32)
	case uint16:
		u32 = uint32(v)
		kind = cFormatUint32
		value = unsafe.Pointer(&u32)
	case uint32:
		u32 = v
		kind = cFormatUint32
		value = unsafe.Pointer(&u32)
	case uint64:
		u64 = v
		kind = cFormatUint64
		value = unsafe.Pointer(&u64)
	case uintptr:
		if unsafe.Sizeof(v) == 8 {
			u64 = uint64(v)
			kind = cFormatUint64
			value = unsafe.Pointer(&u64)
		} else {
			u32 = uint32(v)
			kind = cFormatUint32
			value = unsafe.Pointer(&u32)
		}
	default:
		f.writeString("%!")
		f.writeByte(verb)
		f.writeString("(BADTYPE)")
		return
	}

	cVerb := verb
	if verb == 'd' && cFormatIsUnsigned(kind) {
		// C's %d requires a signed argument. Go's %d supports unsigned values,
		// so use the equivalent C unsigned-decimal verb instead.
		cVerb = 'u'
	} else if verb != 'd' && cFormatIsSigned(kind) {
		// C's x/X/o conversions require an unsigned argument. Preserve the
		// underlying bit pattern while avoiding undefined variadic type usage.
		if kind == cFormatInt64 {
			u64 = uint64(i64)
			kind = cFormatUint64
			value = unsafe.Pointer(&u64)
		} else {
			u32 = uint32(i32)
			kind = cFormatUint32
			value = unsafe.Pointer(&u32)
		}
	}

	var cfmt [64]byte
	if buildCFormat(cfmt[:], cVerb, flags, width, prec, hasWidth, hasPrec, kind) < 0 {
		f.writeCFormatError(verb)
		return
	}

	f.writeCFormatted(verb, kind, unsafe.Pointer(&cfmt[0]), value)
}

// formatFloat handles float formatting verbs (%f, %e, %E, %g, %G) via snprintf.
func (f *formatter) formatFloat(verb byte, flags string, width, prec int, hasWidth, hasPrec bool, arg any) {
	var value float64
	switch v := arg.(type) {
	case float32:
		value = float64(v)
	case float64:
		value = v
	default:
		f.writeString("%!")
		f.writeByte(verb)
		f.writeString("(BADTYPE)")
		return
	}

	var cfmt [64]byte
	if buildCFormat(cfmt[:], verb, flags, width, prec, hasWidth, hasPrec, cFormatDouble) < 0 {
		f.writeCFormatError(verb)
		return
	}

	f.writeCFormatted(
		verb,
		cFormatDouble,
		unsafe.Pointer(&cfmt[0]),
		unsafe.Pointer(&value),
	)
}

// snprintfPtr formats a pointer via snprintf.
func (f *formatter) snprintfPtr(ptr unsafe.Pointer) {
	var cfmt [4]byte
	cfmt[0] = '%'
	cfmt[1] = 'p'
	cfmt[2] = 0

	value := ptr
	f.writeCFormatted(
		'p',
		cFormatPointer,
		unsafe.Pointer(&cfmt[0]),
		unsafe.Pointer(&value),
	)
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
// It returns the number of bytes written excluding the NUL, or -1 if dst is
// too small.
func buildCFormat(dst []byte, verb byte, flags string, width, prec int, hasWidth, hasPrec bool, kind cFormatKind) int {
	if len(dst) < 2 {
		return -1
	}

	i := 0
	dst[i] = '%'
	i++

	for j := 0; j < len(flags); j++ {
		if i >= len(dst)-1 {
			return -1
		}
		dst[i] = flags[j]
		i++
	}

	// A negative width supplied through '*' is equivalent to the '-' flag and
	// a positive width. intMagnitude handles the minimum int without overflow.
	if hasWidth && width < 0 && !containsByte(flags, '-') {
		if i >= len(dst)-1 {
			return -1
		}
		dst[i] = '-'
		i++
	}

	if hasWidth {
		magnitude := intMagnitude(width)
		digits := decimalDigits(magnitude)
		if i+digits >= len(dst) {
			return -1
		}
		i += writeUintToBytes(dst[i:], magnitude)
	}

	// A negative precision supplied through '*' means that precision was not
	// specified.
	if hasPrec && prec >= 0 {
		if i >= len(dst)-1 {
			return -1
		}
		dst[i] = '.'
		i++

		magnitude := uint(prec)
		digits := decimalDigits(magnitude)
		if i+digits >= len(dst) {
			return -1
		}
		i += writeUintToBytes(dst[i:], magnitude)
	}

	if cFormatIs64Bit(kind) {
		if i+2 >= len(dst) {
			return -1
		}
		dst[i] = 'l'
		i++
		dst[i] = 'l'
		i++
	}

	if i >= len(dst)-1 {
		return -1
	}
	dst[i] = verb
	i++
	dst[i] = 0

	return i
}

func containsByte(s string, want byte) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == want {
			return true
		}
	}
	return false
}

func intMagnitude(value int) uint {
	if value >= 0 {
		return uint(value)
	}

	// -(minimum int) overflows. Negate value+1, then add the missing unit in
	// the unsigned domain.
	return uint(-(value + 1)) + 1
}

func decimalDigits(value uint) int {
	digits := 1
	for value >= 10 {
		value /= 10
		digits++
	}
	return digits
}

// writeUintToBytes writes a non-negative integer as decimal digits into dst.
// Returns the number of bytes written.
func writeUintToBytes(dst []byte, value uint) int {
	if value == 0 {
		dst[0] = '0'
		return 1
	}

	var tmp [20]byte
	i := len(tmp)
	for value > 0 {
		i--
		tmp[i] = byte('0' + value%10)
		value /= 10
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
