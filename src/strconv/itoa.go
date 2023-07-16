package strconv

import (
	"unsafe"
)

func FormatInt(i int64, base int) string {
	var result string
	ptr := (*_string)(unsafe.Pointer(&result))

	// Allocate space for the string array
	ptr.array = _alloc(12)

	// Call into the C library
	ptr.array = _itoa(int(i), ptr.array, base)

	// Update the length of the string
	ptr.len = int(_strlen(ptr.array))

	return result
}

func Itoa(i int) string {
	return FormatInt(int64(i), 10)
}
