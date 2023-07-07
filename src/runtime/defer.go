package runtime

import (
	"unsafe"
)

type deferFrame struct {
	fn   unsafe.Pointer
	ctx  unsafe.Pointer
	next *deferFrame
	top  *deferFrame
}

func deferPush(fn unsafe.Pointer, ctx unsafe.Pointer, top **deferFrame) {
	if *top == nil {
		// Initialize the top pointer for the duration of the callee
		*top = deferCurrentTop()
	}

	currentTask.deferStack = &deferFrame{
		fn:   fn,
		ctx:  ctx,
		next: currentTask.deferStack,
		top:  *top,
	}
}

func deferRun(top *deferFrame) {
	lastState := currentTask.state
	for currentTask.deferStack != top && currentTask.deferStack != nil {
		frame := currentTask.deferStack

		// Pop frame from stack
		currentTask.deferStack = frame.next

		// Call the deferred function
		deferCall(frame.fn, frame.ctx)

		// Check if a panic recovered
		if lastState == taskPanicking && currentTask.state == taskRecovered {
			// Transition this task back to the running state
			currentTask.state = taskRunning

			// Stop the panic sequence
			break
		}
	}
}

func deferCurrentTop() *deferFrame {
	return currentTask.deferStack
}

func deferInitialTop() *deferFrame {
	if currentTask.deferStack != nil {
		return currentTask.deferStack.top
	}
	return nil
}

// deferCall is an intrinsic for executing the deferred function
func deferCall(fn unsafe.Pointer, ctx unsafe.Pointer)
