package types

import "unsafe"

type stringDescriptor struct {
	array unsafe.Pointer
	len   int
}

func stringLen(descriptor stringDescriptor) int {
	return 0
}
