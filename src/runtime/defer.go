package runtime

import "unsafe"

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
func setjmp(*jmp_buf) int32

//sigo:extern longjmp longjmp
func longjmp(*jmp_buf, int32)

func deferStackCreate() deferStack {
	return deferStack{}
}

//go:nowritebarrier
func deferInit(isLongjmp int32, stack *deferStack) bool {
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

//go:nowritebarrier
func deferPush(s *deferStack, fn _func) {
	// Push the defer frame to the top of the defer stack for the current function
	ptr := alloc(unsafe.Sizeof(deferFrame{}))
	frame := (*deferFrame)(ptr)
	frame.fn = fn
	frame.next = s.head
	s.head = frame
}

//go:nowritebarrier
func deferRun(s *deferStack) {
	lastState := currentGoroutine.state
	for s.head != nil {
		// Pop a frame from the stack
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
