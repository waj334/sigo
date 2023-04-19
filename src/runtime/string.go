package runtime

import "unsafe"

type stringDescriptor struct {
	array unsafe.Pointer
	len   int
}

//go:export stringLen runtime.stringLen
func stringLen(descriptor unsafe.Pointer) int {
	return 0
}

//go:export stringCompare runtime.stringCompare
func stringCompare(lhs, rhs unsafe.Pointer) bool {
	// TODO: This function can just call strncmp from the standard C library
	//  when it is able to be called.
	strLhs := *(*stringDescriptor)(lhs)
	strRhs := *(*stringDescriptor)(rhs)

	// Fast path
	if strLhs.len != strRhs.len {
		return false
	}

	// Compare each byte
	for i := 0; i < strLhs.len; i++ {
		charLhs := *(*byte)(unsafe.Pointer(uintptr(strLhs.array) + uintptr(i)))
		charRhs := *(*byte)(unsafe.Pointer(uintptr(strRhs.array) + uintptr(i)))
		if charLhs != charRhs {
			return false
		}
	}
	return true
}
