package runtime

import "unsafe"

//sigo:export runtime_registerWeakPointer runtime_registerWeakPointer
func runtime_registerWeakPointer(ptr unsafe.Pointer) unsafe.Pointer {
	return ptr
}

//sigo:export runtime_makeStrongFromWeak runtime_makeStrongFromWeak
func runtime_makeStrongFromWeak(ptr unsafe.Pointer) unsafe.Pointer {
	return ptr
}
