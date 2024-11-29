package runtime

import "unsafe"

func _panic(arg any) {
	// Transition the current task to the panicking state
	currentTask.state = taskPanicking

	// Store the arg in the current goroutine's context
	currentTask.panicValue = arg

	// TODO: Attempt to print the arguments

	if currentTask.deferStack != nil {
		// Begin unwinding the stack.
		longjmp(&currentTask.deferStack.jb, 1)
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

type exception struct {
	value      any
	deferStack *deferStack
}

func goPersonality(version int32, actions int32, class uint64, e *exception, ctx unsafe.Pointer) int32 {
	if actions&1 != 0 {
		// This is a cleanup action.
		if e.deferStack != nil {
			deferRun(e.deferStack)
		}
		return 1
	}
	return 0
}
