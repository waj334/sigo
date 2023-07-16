package strconv

import "unsafe"

//sigo:extern _alloc runtime.alloc
//sigo:extern _itoa itoa
//sigo:extern _strlen strlen

type _string struct {
	array unsafe.Pointer
	len   int
}

func _alloc(size uintptr) unsafe.Pointer
func _itoa(value int, str unsafe.Pointer, base int) unsafe.Pointer
func _strlen(pointer unsafe.Pointer) uintptr
