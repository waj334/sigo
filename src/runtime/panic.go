package runtime

//go:export panicHandler runtime.panic
func panicHandler(arg any) {
	// TODO: Attempt to print the arguments
	// TODO: Call the current function's defer stack

	// Abort if not recovered
	abort()
}
