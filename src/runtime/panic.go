package runtime

func _panic(arg any) {
	// Transition the current task to the panicking state
	currentTask.state = taskPanicking

	// Store the arg in the current goroutine's context
	currentTask.panicValue = arg

	// TODO: Attempt to print the arguments

	// Call the current goroutine's defer stack
	deferRun(nil)

	// Abort if not recovered
	if currentTask.state == taskPanicking {
		abort()
	}
}

func _recover() any {
	if currentTask.state == taskPanicking {
		// Transition task state to recovered
		currentTask.state = taskRecovered

		// Return the argument passed to panic
		return currentTask.panicValue
	}
	return nil
}
