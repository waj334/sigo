package runtime

import (
	"os"
	"unsafe"
)

// _print writes the textual representation of each argument to os.Stdout, with
// no separators. It is the lowering target of the Go `print` builtin.
//
// Only types that the runtime can reasonably format without pulling in fmt are
// supported: bool, signed/unsigned integers, uintptr, string, []byte,
// unsafe.Pointer. Anything else is rendered as "?<type>?".
func _print(args ...any) {
	for _, a := range args {
		printOne(a)
	}
}

// _println is _print followed by a newline. It does not insert separators
// between args (matching Go's `print`/`println` builtins, which only add a
// trailing newline for `println`).
func _println(args ...any) {
	for _, a := range args {
		printOne(a)
	}
	if os.Stdout != nil {
		_, _ = os.Stdout.Write([]byte{'\n'})
	}
}

func printOne(a any) {
	if os.Stdout == nil {
		return
	}
	switch v := a.(type) {
	case string:
		_, _ = os.Stdout.Write([]byte(v))
	case []byte:
		_, _ = os.Stdout.Write(v)
	case bool:
		if v {
			_, _ = os.Stdout.Write([]byte("true"))
		} else {
			_, _ = os.Stdout.Write([]byte("false"))
		}
	case int:
		writeInt64(int64(v))
	case int8:
		writeInt64(int64(v))
	case int16:
		writeInt64(int64(v))
	case int32:
		writeInt64(int64(v))
	case int64:
		writeInt64(v)
	case uint:
		writeUint64(uint64(v))
	case uint8:
		writeUint64(uint64(v))
	case uint16:
		writeUint64(uint64(v))
	case uint32:
		writeUint64(uint64(v))
	case uint64:
		writeUint64(v)
	case uintptr:
		writeHex64(uint64(v))
	case unsafe.Pointer:
		writeHex64(uint64(uintptr(v)))
	default:
		_, _ = os.Stdout.Write([]byte("?"))
	}
}

func writeInt64(v int64) {
	if v < 0 {
		_, _ = os.Stdout.Write([]byte{'-'})
		v = -v
	}
	writeUint64(uint64(v))
}

func writeUint64(v uint64) {
	var buf [20]byte
	i := len(buf)
	if v == 0 {
		_, _ = os.Stdout.Write([]byte{'0'})
		return
	}
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	_, _ = os.Stdout.Write(buf[i:])
}

func writeHex64(v uint64) {
	const hex = "0123456789abcdef"
	var buf [18]byte
	buf[0] = '0'
	buf[1] = 'x'
	i := len(buf)
	if v == 0 {
		_, _ = os.Stdout.Write([]byte("0x0"))
		return
	}
	for v > 0 {
		i--
		buf[i] = hex[v&0xf]
		v >>= 4
	}
	out := append([]byte("0x"), buf[i:]...)
	_, _ = os.Stdout.Write(out)
}
