package _go

import "unsafe"

type interfaceDescriptor struct {
	typePtr  unsafe.Pointer
	valuePtr unsafe.Pointer
}

//go:linkname runtime.makeInterface
func interfaceMake(value unsafe.Pointer, valueType unsafe.Pointer) interfaceDescriptor {
	return interfaceDescriptor{
		typePtr:  valueType,
		valuePtr: value,
	}
}

//go:linkname runtime.typeAssert
func interfaceAssert(X unsafe.Pointer, from unsafe.Pointer, to unsafe.Pointer) (unsafe.Pointer, bool) {
	return nil, false
}
