package runtime

import (
	"unsafe"
)

type deferStack struct {
	head *deferFrame
	next *deferStack
}

type deferFrame struct {
	fn   unsafe.Pointer
	ctx  unsafe.Pointer
	next *deferFrame
}

func deferStartStack() {
	currentTask.deferStack = &deferStack{
		head: nil,
		next: currentTask.deferStack,
	}
}

func deferPush(fn unsafe.Pointer, ctx unsafe.Pointer) {
	// Push the defer frame to the top of the defer stack for the current function
	currentTask.deferStack.head = &deferFrame{
		fn:   fn,
		ctx:  ctx,
		next: currentTask.deferStack.head,
	}
}

func deferRun() {
	lastState := currentTask.state
	for currentTask.deferStack != nil {
		for currentTask.deferStack.head != nil {
			frame := currentTask.deferStack.head

			// Pop frame from stack
			currentTask.deferStack.head = frame.next

			// Dispatch the deferred function
			_dispatch(frame.fn, frame.ctx)

			// Check if a panic recovered
			if lastState == taskPanicking && currentTask.state == taskRecovered {
				// Transition this task back to the running state
				currentTask.state = taskRunning
			}
		}

		if currentTask.state == taskPanicking {
			// Begin executing the next defer stack
			currentTask.deferStack = currentTask.deferStack.next
		} else {
			break
		}
	}
}
