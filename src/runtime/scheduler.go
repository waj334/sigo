package runtime

import (
	"internal/chacha8rand"
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
	goroutineExiting
)

type _func struct {
	f         unsafe.Pointer
	args      unsafe.Pointer
	stackSize uintptr
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
	chacha8    chacha8rand.State
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
	headGoroutine      *goroutine
	lastGoroutine      *goroutine
	currentGoroutine   *goroutine
	targetGoroutine    *goroutine
	retiredGoroutines  *goroutine
	goroutineStackSize uintptr
)

func initGoroutine(unsafe.Pointer)
func alignStack(n uintptr) uintptr
func gosched()

func schedule() bool {
	// This is the goroutine whose context is physically present on PSP.
	// Do not infer this from lastGoroutine; that value may describe an older
	// context switch.
	outgoing := currentGoroutine

	if headGoroutine == nil {
		targetGoroutine = nil
		lastGoroutine = outgoing
		return false
	}

	// Do not switch during panic handling. An exiting goroutine is deliberately
	// not included here: it must be allowed to switch away permanently.
	if outgoing != nil {
		switch outgoing.state {
		case goroutinePanicking, goroutineRecovered:
			lastGoroutine = outgoing
			return false
		}
	}

	var next *goroutine

	if outgoing == nil {
		// First scheduler activation. Search the ring for the first runnable
		// goroutine rather than assuming headGoroutine is runnable.
		start := headGoroutine
		candidate := start

		for {
			if runnable(candidate) {
				next = candidate
				break
			}

			candidate = candidate.next
			if candidate == start {
				break
			}
		}
	} else if targetGoroutine != nil {
		// Consume the targeted wakeup exactly once.
		target := targetGoroutine
		targetGoroutine = nil

		if target == outgoing {
			// The goroutine was resumed after marking itself parked but before
			// PendSV actually switched away. Its context is still physically
			// active on PSP.
			if outgoing.state == goroutineReady {
				outgoing.state = goroutineRunning
			}

			if outgoing.state == goroutineRunning {
				lastGoroutine = outgoing
				return false
			}

			// If outgoing is parked or exiting, ignore this stale target and
			// continue with normal ring selection.
		} else if runnable(target) {
			next = target
		}
	}

	if next == nil && outgoing != nil {
		// Search the ring exactly once, starting after the outgoing goroutine.
		// Do not select parked, panicking, recovered, running, or exiting
		// goroutines.
		candidate := outgoing.next
		start := candidate

		for {
			if candidate != outgoing && runnable(candidate) {
				next = candidate
				break
			}

			candidate = candidate.next
			if candidate == start {
				break
			}
		}

		// The outgoing goroutine may remain selected only if it is still
		// runnable. An exiting goroutine must never be selected again.
		if next == nil &&
			(outgoing.state == goroutineRunning ||
				outgoing.state == goroutineReady) {

			next = outgoing
		}
	}

	if next == nil || next == outgoing {
		// No actual context change occurred.
		if outgoing != nil && outgoing.state == goroutineReady {
			outgoing.state = goroutineRunning
		}

		lastGoroutine = outgoing
		return false
	}

	// Commit the context-switch decision. PendSV will save lastGoroutine's
	// active PSP context and restore currentGoroutine's saved context.
	lastGoroutine = outgoing
	currentGoroutine = next

	if outgoing != nil {
		switch outgoing.state {
		case goroutineRunning:
			outgoing.state = goroutineReady

		case goroutineExiting:
			// It is now safe to remove the outgoing goroutine from the ring,
			// but not to free its stack. PendSV still needs the stack to save
			// the outgoing hardware/software context before restoring next.
			unlinkGoroutine(outgoing)
			retireGoroutine(outgoing)
		}
	}

	next.state = goroutineRunning
	return true
}

func addGoroutine(f _func) {
	if f.f == nil {
		// Do nothing.
		return
	}

	state := DisableInterrupts()

	// Allocate stack for this goroutine.
	stackSize := f.stackSize
	if stackSize == 0 {
		stackSize = goroutineStackSize
	}

	//stack := alloc(stackSize)
	stack := malloc(stackSize)

	// Create the new goroutine
	newGoroutine := &goroutine{
		stack: stack,
		// NOTE: initGoroutine may move the top of the stack pointer depending on the target machine's stack growth
		//       direction.
		stackTop: stack,
		__func: _func{
			f:         f.f,
			args:      f.args,
			stackSize: stackSize,
		},
		state: goroutineNotStarted,
	}

	// Seed chacha8 with the goroutine's address.
	gint := uintptr(unsafe.Pointer(newGoroutine))
	newGoroutine.chacha8.Init64([4]uint64{
		uint64(gint >> 8),
		uint64(gint >> 16),
		uint64(gint >> 24),
		uint64(gint >> 32),
	})

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

// removeGoroutine terminates the currently executing goroutine.
//
// It cannot synchronously unlink or free g because this function itself is
// executing on g's stack. schedule performs the unlink only after selecting
// another goroutine, and PendSV then switches away from this stack.
//
// This function intentionally never returns.
func removeGoroutine(ptr unsafe.Pointer) {
	g := (*goroutine)(ptr)

	state := DisableInterrupts()

	if g != currentGoroutine {
		EnableInterrupts(state)
		panic("cannot remove a goroutine that is not currently running")
	}

	g.state = goroutineExiting

	if targetGoroutine == g {
		targetGoroutine = nil
	}

	EnableInterrupts(state)

	// Keep requesting a switch. If every other goroutine is parked, this
	// remains here until an interrupt wakes one. Once a switch succeeds, this
	// goroutine is unlinked and can never resume.
	for {
		gosched()
	}
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

	if g.state != goroutineParked {
		EnableInterrupts(state)
		return
	}

	// The goroutine was woken after marking itself parked but before PendSV
	// actually switched away from it. It is still the active context.
	if g == currentGoroutine {
		g.state = goroutineRunning
		EnableInterrupts(state)
		return
	}

	g.state = goroutineReady
	targetGoroutine = g
	EnableInterrupts(state)

	gosched()
}

func runnable(g *goroutine) bool {
	if g == nil {
		return false
	}

	return g.state == goroutineNotStarted ||
		g.state == goroutineReady
}

func getg() *goroutine {
	return currentGoroutine
}

// getgPtr returns the current goroutine as an unsafe.Pointer. It exists so
// callers in the time and sync packages can hold an opaque goroutine handle
// without referencing the runtime's *goroutine type, which is private to the
// runtime. The two getg variants would otherwise have signatures that differ
// only in their return type, which would mismatch at the func.call site
// when the time/sync side is bridged via //sigo:extern.
//
//go:export getgPtr runtime.getgPtr
func getgPtr() unsafe.Pointer {
	return unsafe.Pointer(currentGoroutine)
}

// unlinkGoroutine removes g from the runnable ring.
//
// This must only be called by schedule after it has selected a different
// incoming goroutine. The outgoing stack is still active until PendSV finishes,
// so this function does not free anything.
func unlinkGoroutine(g *goroutine) {
	if g.next == g {
		headGoroutine = nil
	} else {
		g.prev.next = g.next
		g.next.prev = g.prev

		if headGoroutine == g {
			headGoroutine = g.next
		}
	}

	if targetGoroutine == g {
		targetGoroutine = nil
	}

	g.next = nil
	g.prev = nil
}

// retireGoroutine queues an unlinked goroutine for later reclamation.
//
// schedule runs from PendSV, so it must not invoke the normal allocator or
// free the stack there. The retired list is consumed later in Thread mode.
func retireGoroutine(g *goroutine) {
	g.next = retiredGoroutines
	retiredGoroutines = g
}
