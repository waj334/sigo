package runtime

//go:export panicHandler runtime._panic
func _panic(arg any) {
	// TODO: Attempt to print the arguments
	// TODO: Call the current function's defer stack

	// Abort if not recovered
	abort()
}
