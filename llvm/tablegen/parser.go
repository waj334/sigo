package tablegen

// #include "tablegen.h"
// #include <stdlib.h>
import "C"

import (
	"unsafe"
)

func ParseTableGenFile(path string, rk *RecordKeeper, includes []string) bool {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	cIncludes, cleanup := cstringArray(includes)
	defer cleanup()

	return bool(C.LLVMTableGenParseFile(
		cpath,
		rk.Ptr(),
		cIncludes,
		C.size_t(len(includes)),
	))
}
