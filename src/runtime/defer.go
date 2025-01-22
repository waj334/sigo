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
		nextStack := currentGoroutine.deferStack.next

		// Execute the defers.
		deferRun(currentGoroutine.deferStack)

		// Unwind the stack.
		if currentGoroutine.state == goroutinePanicking {
			if nextStack != nil {
				longjmp(&nextStack.jb, 1)
			} else {
				// Unrecovered panic
				abort()
			}
		}
		return true
	} else {
		stack.next = currentGoroutine.deferStack
		currentGoroutine.deferStack = stack
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
	lastState := currentGoroutine.state
	for s.head != nil {
		// Pop frame from stack
		frame := s.head
		s.head = frame.next

		// Execute the deferred function
		exec(frame.fn.args, frame.fn.f)

		// Check if a panic recovered
		if lastState == goroutinePanicking && currentGoroutine.state == goroutineRecovered {
			// Transition this goroutine back to the running state
			currentGoroutine.state = goroutineRunning
		}
	}

	// Pop this defer stack.
	currentGoroutine.deferStack = s.next
}
