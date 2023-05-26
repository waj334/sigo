package cortexm

/*
type exceptionStack struct {
	R0  uintptr
	R1  uintptr
	R2  uintptr
	R3  uintptr
	R12 uintptr
	LR  uintptr
	PC  uint
	PSR uintptr
}*/

//go:export _hardfault _hardfault
func _hardfault(estack exceptionStack) {
	abort()
}
