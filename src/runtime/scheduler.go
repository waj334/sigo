package runtime

import (
	"unsafe"
)

type goroutineState uint8

const (
	goroutineNotStarted goroutineState = iota
	goroutineReady
	goroutineRunning
	goroutinePanicking
	goroutineRecovered
	goroutineParked
)

type _func struct {
	f    unsafe.Pointer
	args unsafe.Pointer
}

type goroutine struct {
	stackTop   unsafe.Pointer
	__func     _func
	stack      unsafe.Pointer
	next       *goroutine
	prev       *goroutine
	state      goroutineState
	deferStack *deferStack
	panicValue any
}

//sigo:extern goroutineStackSize runtime._goroutineStackSize
//sigo:extern initGoroutine runtime.initGoroutine
//sigo:extern alignStack runtime.alignStack
//sigo:extern gosched runtime.gosched

//sigo:export headGoroutine runtime.headGoroutine
//go:export lastGoroutine runtime.lastGoroutine
//go:export currentGoroutine runtime.currentGoroutine
//go:export schedule runtime.schedule
//go:export addGoroutine runtime.addGoroutine
//go:export removeGoroutine runtime.removeGoroutine
//go:export sleep runtime.sleep
//go:export gopark runtime.gopark
//go:export goresume runtime.goresume
//go:export getg runtime.getg

var (
	headGoroutine      *goroutine = nil
	lastGoroutine      *goroutine = nil
	currentGoroutine   *goroutine = nil
	targetGoroutine    *goroutine = nil
	goroutineStackSize uintptr
)

func initGoroutine(unsafe.Pointer)
func alignStack(n uintptr) uintptr
func gosched()

func schedule() bool {
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
		} else if targetGoroutine != nil {
			// Switch to the target goroutine.
			lastGoroutine = currentGoroutine
			currentGoroutine = targetGoroutine
			targetGoroutine = nil
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

			start := nextGoroutine
			for {
				if nextGoroutine.state != goroutineParked {
					currentGoroutine = nextGoroutine
					break
				}
				nextGoroutine = nextGoroutine.next
				if nextGoroutine == start {
					// All goroutines are parked.
					EnableInterrupts(state)
					return false
				}
			}
		}

		if currentGoroutine != nil && currentGoroutine != lastGoroutine && currentGoroutine.state != goroutineRunning {
			if lastGoroutine != nil && lastGoroutine.state == goroutineRunning {
				// Transition the last goroutine to the ready state.
				lastGoroutine.state = goroutineReady
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
		// NOTE: initGoroutine may move the top of the stack pointer depending on the target machine's stack growth
		//       direction.
		stackTop: stack,
		__func:   f,
		state:    goroutineNotStarted,
	}

	// Initialize the stack for this goroutine.
	initGoroutine(unsafe.Pointer(newGoroutine))

	// Insert into the goroutine ring.
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

func gopark(ptr unsafe.Pointer) {
	state := DisableInterrupts()
	g := (*goroutine)(ptr)
	g.state = goroutineParked
	EnableInterrupts(state)
	for g.state == goroutineParked {
		gosched()
	}
}

// goparkWithCallback parks the goroutine and executes a callback atomically
// after marking as parked but before enabling interrupts. This prevents races
// where an interrupt could fire between parking and the callback execution.
//
//go:export goparkWithCallback runtime.goparkWithCallback
func goparkWithCallback(ptr unsafe.Pointer, callback func()) {
	state := DisableInterrupts()
	g := (*goroutine)(ptr)
	g.state = goroutineParked
	// Execute callback while interrupts are still disabled
	// This ensures atomicity between parking and callback
	if callback != nil {
		callback()
	}
	EnableInterrupts(state)
	for g.state == goroutineParked {
		gosched()
	}
}

func goresume(ptr unsafe.Pointer) {
	state := DisableInterrupts()
	g := (*goroutine)(ptr)
	if g.state == goroutineParked {
		g.state = goroutineReady
		EnableInterrupts(state)

		targetGoroutine = g
		gosched()
		return
	}
	EnableInterrupts(state)
}

func getg() unsafe.Pointer {
	return unsafe.Pointer(currentGoroutine)
}
