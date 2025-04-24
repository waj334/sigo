package runtime

import (
	"time"
	"unsafe"
)

type goroutineState uint8

const (
	goroutineNotStarted goroutineState = iota
	goroutineIdle
	goroutineSleep
	goroutineRunning
	goroutinePanicking
	goroutineRecovered
	goroutineWaiting
)

type _func struct {
	f    unsafe.Pointer
	args unsafe.Pointer
}

type goroutine struct {
	stackTop      unsafe.Pointer
	__func        _func
	stack         unsafe.Pointer
	next          *goroutine
	prev          *goroutine
	state         goroutineState
	sleepDeadline uint64
	deferStack    *deferStack
	panicValue    any
}

//sigo:extern goroutineStackSize runtime._goroutineStackSize
//sigo:extern initGoroutine runtime.initGoroutine
//sigo:extern alignStack runtime.alignStack
//sigo:extern schedulerPause runtime.schedulerPause

//go:export lastGoroutine runtime.lastGoroutine
//go:export currentGoroutine runtime.currentGoroutine
//go:export runScheduler runtime.runScheduler
//go:export addGoroutine runtime.addGoroutine
//go:export removeGoroutine runtime.removeGoroutine
//go:export sleep runtime.sleep
//go:export waitGoroutine runtime.waitGoroutine
//go:export resumeGoroutine runtime.resumeGoroutine
//go:export runningGoroutine runtime.runningGoroutine

//sigo:required runScheduler

var (
	headGoroutine      *goroutine = nil
	lastGoroutine      *goroutine = nil
	currentGoroutine   *goroutine = nil
	goroutineStackSize uintptr
)

func initGoroutine(unsafe.Pointer)
func alignStack(n uintptr) uintptr
func schedulerPause()

func runScheduler() bool {
	state := DisableInterrupts()

	// Check for stack overflow on current goroutine.
	if currentGoroutine != nil {
		stackBottom := unsafe.Add(currentGoroutine.stack, goroutineStackSize)
		stackSize := uintptr(stackBottom) - uintptr(currentStack())
		if stackSize > goroutineStackSize {
			panic("stack overflow")
		}
	}

	if headGoroutine != nil {
		if currentGoroutine == nil {
			// Initialize the current goroutine.
			currentGoroutine = headGoroutine
			lastGoroutine = nil
		} else {
			if currentGoroutine.state == goroutinePanicking || currentGoroutine.state == goroutineRecovered {
				// Do not allow any further context switches from this goroutine.
				// NOTE: Interrupts are intentionally not re-enabled. The panic will re-enable them if a panic is
				//		 recovered.
				lastGoroutine = currentGoroutine
				return false
			}

			// Switch to the next goroutine
			lastGoroutine = currentGoroutine
			nextGoroutine := lastGoroutine.next

			for {
				if nextGoroutine.state == goroutineSleep {
					t := uint64(time.Now().UnixNano())
					if t > nextGoroutine.sleepDeadline {
						nextGoroutine.state = goroutineIdle
						nextGoroutine.sleepDeadline = 0
					} else if nextGoroutine == lastGoroutine && nextGoroutine.state == goroutineSleep {
						// All goroutines are sleep. panic
						panic("all goroutines are sleep")
					} else {
						// Skip sleeping goroutine
						nextGoroutine = nextGoroutine.next
						continue
					}
				} else if nextGoroutine.state == goroutineWaiting {
					// skip waiting goroutines
					nextGoroutine = nextGoroutine.next
					continue
				}
				currentGoroutine = nextGoroutine
				break
			}
		}

		if currentGoroutine != nil && currentGoroutine != lastGoroutine && currentGoroutine.state != goroutineRunning {
			if lastGoroutine != nil && lastGoroutine.state == goroutineRunning {
				// Transition the last goroutine to the idle state.
				lastGoroutine.state = goroutineIdle
			}

			// Transition the new current goroutine to the running state.
			currentGoroutine.state = goroutineRunning

			// Re-enable interrupts and signal that a context switch to the new goroutine must take place.
			EnableInterrupts(state)
			return true
		}
	}

	EnableInterrupts(state)

	// Signal that no context switch should occur.
	return false
}

func addGoroutine(f _func) {
	if f.f == nil {
		// Do nothing.
		return
	}

	state := DisableInterrupts()

	// Allocate stack for this goroutine.
	stackSize := goroutineStackSize
	stack := alloc(stackSize)

	// Create the new goroutine
	newGoroutine := &goroutine{
		stack: stack,
		// initGoroutine may move the top of stack pointer depending on the target machine's stack growth direction.
		stackTop: stack,
		__func:   f,
		state:    goroutineNotStarted,
	}

	// Initialize the stack for this goroutine.
	initGoroutine(unsafe.Pointer(newGoroutine))

	// Insert into ring
	oldHead := headGoroutine
	headGoroutine = newGoroutine
	if oldHead == nil {
		headGoroutine.next = headGoroutine
		headGoroutine.prev = headGoroutine
	} else {
		// Insert the new goroutine before the old head goroutine.
		headGoroutine.next = oldHead
		headGoroutine.prev = oldHead.prev

		oldHead.prev.next = headGoroutine
		oldHead.prev = headGoroutine
	}
	EnableInterrupts(state)
}

func removeGoroutine(ptr unsafe.Pointer) {
	state := DisableInterrupts()
	g := (*goroutine)(ptr)

	// Free this goroutine's stack.
	free(g.stack)

	if g.next == g && g.prev == g {
		// There is only one goroutine left.
		headGoroutine = nil
		currentGoroutine = nil
		lastGoroutine = nil
	} else {
		// Remove the goroutine from the ring.
		g.prev.next = g.next
		g.next.prev = g.prev

		// Advance to the next goroutine.
		if g == currentGoroutine {
			currentGoroutine = g.prev
		}

		// If the goroutine being removed was the head goroutine, set the next goroutine as the new head goroutine.
		if g == headGoroutine {
			headGoroutine = g.next
		}

		// If the goroutine being removed was the current goroutine, set the next goroutine as the new current goroutine.
		if g == currentGoroutine {
			currentGoroutine = g.next
		}

		// If the goroutine being removed was the last goroutine, set the previous goroutine as the new last goroutine.
		if g == lastGoroutine {
			lastGoroutine = g.prev
		}
	}

	EnableInterrupts(state)
}

func waitGoroutine(ptr unsafe.Pointer) {
	g := (*goroutine)(ptr)
	if g.state != goroutineWaiting {
		state := DisableInterrupts()
		g.state = goroutineWaiting
		EnableInterrupts(state)

		// Schedule another goroutine to begin running.
		schedulerPause()
	}
}

func resumeGoroutine(ptr unsafe.Pointer) {
	g := (*goroutine)(ptr)
	if g.state == goroutineWaiting {
		state := DisableInterrupts()
		g.state = goroutineIdle
		EnableInterrupts(state)
	}
}

func runningGoroutine() unsafe.Pointer {
	return unsafe.Pointer(currentGoroutine)
}

func sleep(d uint64) {
	if currentGoroutine == nil {
		panic("sleep called from non-goroutine")
	}
	currentGoroutine.sleepDeadline = uint64(time.Now().UnixNano()) + d
	currentGoroutine.state = goroutineSleep

	// Schedule another goroutine to begin running.
	schedulerPause()
}
