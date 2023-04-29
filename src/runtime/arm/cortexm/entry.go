package cortexm

import "unsafe"

//go:linkname main main.main
func main()

//sigo:extern _start_bss _start_bss
var _start_bss unsafe.Pointer

//sigo:extern _end_bss _end_bss
var _end_bss unsafe.Pointer

//sigo:extern _start_data _start_data
var _start_data unsafe.Pointer

//sigo:extern _end_data _end_data
var _end_data unsafe.Pointer

//sigo:extern _data_base_addr _data_base_addr
var _data_base_addr unsafe.Pointer

func initMemory() {
	// Zero init globals
	sbss := unsafe.Pointer(&_start_bss)
	ebss := unsafe.Pointer(&_end_bss)
	for ptr := sbss; ptr != ebss; ptr = unsafe.Add(ptr, 4) {
		*(*uint32)(ptr) = 0
	}

	// Initialize data from flash
	dst := unsafe.Pointer(&_start_data)
	src := unsafe.Pointer(&_data_base_addr)
	edata := unsafe.Pointer(&_end_data)
	for dst != edata {
		*(*uint32)(dst) = *(*uint32)(src)
		dst = unsafe.Add(dst, 4)
		src = unsafe.Add(src, 4)
	}
}

//go:export _entry Reset_Handler
func _entry() {
	// Initialize the global variables
	initMemory()

	// Run the main program
	main()
}
