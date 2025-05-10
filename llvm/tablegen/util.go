package tablegen

// #include <stdlib.h>
import "C"

import (
	"unsafe"
)

func cstringArray(strs []string) (**C.char, func()) {
	if len(strs) == 0 {
		return nil, func() {}
	}
	cStrs := make([]*C.char, len(strs))
	for i, s := range strs {
		cStrs[i] = C.CString(s)
	}

	cleanup := func() {
		for _, s := range cStrs {
			C.free(unsafe.Pointer(s))
		}
	}

	return (**C.char)(unsafe.Pointer(&cStrs[0])), cleanup
}
