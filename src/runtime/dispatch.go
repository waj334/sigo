package runtime

import "unsafe"

//sigo:extern _dispatch _dispatch
func _dispatch(fnPtr unsafe.Pointer, args unsafe.Pointer) unsafe.Pointer
