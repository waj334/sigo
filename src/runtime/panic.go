package runtime

import "unsafe"

func _panic(arg any) {
	// Transition the current goroutine to the panicking state
	currentGoroutine.state = goroutinePanicking

	// Store the arg in the current goroutine's context
	currentGoroutine.panicValue = arg

	// TODO: Attempt to print the arguments

	if currentGoroutine.deferStack != nil {
		// Begin unwinding the stack.
		longjmp(&currentGoroutine.deferStack.jb, 1)
	}

	for currentGoroutine.state == goroutinePanicking {
		abort()
	}
}

func _recover() any {
	if currentGoroutine.state == goroutinePanicking {
		// Transition goroutine state to recovered
		currentGoroutine.state = goroutineRecovered

		// Return the argument passed to panic
		return currentGoroutine.panicValue
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
