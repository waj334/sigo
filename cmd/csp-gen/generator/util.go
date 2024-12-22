package generator

import "fmt"

func Mask(bits, offset uintptr) uintptr {
	var result uintptr
	for i := uintptr(0); i < bits; i++ {
		result |= 1 << i
	}
	return result << offset
}

func NextPow2(n uintptr) uintptr {
	v := n
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v++
	return v
}

func DataType(width uintptr) string {
	if width == 1 {
		return "bool"
	} else {
		return fmt.Sprintf("uint%d", max(8, NextPow2(width)))
	}
}
