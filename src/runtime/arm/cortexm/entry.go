package cortexm

import (
	"unsafe"
)

//go:linkname main main.main
func main()

//go:linkname abort _abort
func abort()

//go:linkname initPackages runtime.initPackages
func initPackages()

//go:linkname initSysTick initSysTick
func initSysTick()

//sigo:extern __start_bss __start_bss
var __start_bss unsafe.Pointer

//sigo:extern __end_bss __end_bss
var __end_bss unsafe.Pointer

//sigo:extern __start_data __start_data
var __start_data unsafe.Pointer

//sigo:extern __end_data __end_data
var __end_data unsafe.Pointer

//sigo:extern __data_base_addr __data_base_addr
var __data_base_addr unsafe.Pointer

func initMemory() {
	// Zero init globals
	sbss := unsafe.Pointer(&__start_bss)
	ebss := unsafe.Pointer(&__end_bss)
	for ptr := sbss; ptr != ebss; ptr = unsafe.Add(ptr, 4) {
		*(*uint32)(ptr) = 0
	}

	// Initialize data from flash
	dst := unsafe.Pointer(&__start_data)
	src := unsafe.Pointer(&__data_base_addr)
	edata := unsafe.Pointer(&__end_data)
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

	// Call all the package inits before anything else!
	// NOTE: The package init also sets up the pointer values in the chip support packages
	initPackages()

	// Start the SysTick counter
	initSysTick()

	// Enable interrupts
	EnableInterrupts()

	// Run the main program
	go func() {
		main()
		abort()
	}()

	// Loop forever
	for {
	}
}
