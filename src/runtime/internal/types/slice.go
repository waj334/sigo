package types

import "unsafe"

type sliceDescriptor struct {
	array unsafe.Pointer
	len   int
	cap   int
}

func sliceMake() sliceDescriptor {
	return sliceDescriptor{}
}

func sliceAppend(buf sliceDescriptor, elems sliceDescriptor) sliceDescriptor {
	return sliceDescriptor{}
}

//go:linkname runtime.sliceLen
func sliceLen(s sliceDescriptor) int {
	return s.len
}

func sliceCap(s sliceDescriptor) int {
	return 0
}

func sliceCopy(dst sliceDescriptor) int {
	return 0
}
