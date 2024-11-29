package runtime

type deferStack struct {
	head *deferFrame
	next *deferStack
	jb   jmp_buf
}

type deferFrame struct {
	fn   _func
	next *deferFrame
}

type jmp_buf [16]int

//sigo:extern setjmp setjmp
func setjmp(*jmp_buf) int

//sigo:extern longjmp longjmp
func longjmp(*jmp_buf, int)

func deferStackCreate() deferStack {
	return deferStack{}
}

func deferInit(isLongjmp int, stack *deferStack) bool {
	if isLongjmp != 0 {
		nextStack := currentTask.deferStack.next

		// Execute the defers.
		deferRun(currentTask.deferStack)

		// Unwind the stack.
		if currentTask.state == taskPanicking {
			if nextStack != nil {
				longjmp(&nextStack.jb, 1)
			} else {
				// Unrecovered panic
				abort()
			}
		}
		return true
	} else {
		stack.next = currentTask.deferStack
		currentTask.deferStack = stack
		return false
	}
}

func deferPush(s *deferStack, fn _func) {
	// Push the defer frame to the top of the defer stack for the current function
	s.head = &deferFrame{
		fn:   fn,
		next: s.head,
	}
}

func deferRun(s *deferStack) {
	lastState := currentTask.state
	for s.head != nil {
		// Pop frame from stack
		frame := s.head
		s.head = frame.next

		// Execute the deferred function
		exec(frame.fn.args, frame.fn.f)

		// Check if a panic recovered
		if lastState == taskPanicking && currentTask.state == taskRecovered {
			// Transition this task back to the running state
			currentTask.state = taskRunning
		}
	}

	// Pop this defer stack.
	currentTask.deferStack = s.next
}
