package runtime

import "unsafe"

type mapDescriptor struct {
	values int
}

func mapMake() unsafe.Pointer {
	return nil
}

//go:export runtime.mapLen
func mapLen(m unsafe.Pointer) int {
	return 0
}
