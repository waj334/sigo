package _go

import "unsafe"

type mapDescriptor struct {
	values int
}

func mapMake() unsafe.Pointer {
	return nil
}

//go:linkname runtime.mapLen
func mapLen(m unsafe.Pointer) int {
	return 0
}
