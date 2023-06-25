package runtime

import (
	"sync"
	"unsafe"
)

type _channel struct {
	buffer unsafe.Pointer
	mutex  sync.Mutex
	closed bool
}

func channelMake() _channel {
	return _channel{}
}
