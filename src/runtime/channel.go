package runtime

import "unsafe"

type channelDescriptor struct {
	// TODO
}

func channelMake() unsafe.Pointer {
	return unsafe.Pointer(&channelDescriptor{})
}
