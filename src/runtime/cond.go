package runtime

import "unsafe"

// Condition-variable support for the sync package.
//
// The lost-wakeup hazard with condition variables is the window between
// "release the predicate lock" and "park". These primitives close it by
// splitting the wait into two halves that execute inside one
// interrupts-disabled region on this single-core runtime:
//
//	state := condEnqueue(addr)  // queue current goroutine, interrupts stay off
//	L.Unlock()                  // release predicate lock, still atomic
//	condPark(addr, state)       // mark parked, then re-enable and wait
//
// A waker can only run after condPark has both queued and parked the waiter,
// so condWake always observes a parked, queued goroutine and a wakeup can
// never be dropped.
//
// Waiters share the mutexRoots hash queues with mutex waiters; the two never
// collide because a queue is identified by the exact waitAddr.

// condEnqueue registers the current goroutine as a waiter on the condition
// variable identified by addr. It returns with interrupts still disabled so
// the caller can release its predicate lock and park without a wakeup window.
// The returned state must be passed to condPark, which restores it.
//
//go:export condEnqueue runtime.condEnqueue
func condEnqueue(addr *uint32) uint32 {
	state := DisableInterrupts()

	if InInterrupt() {
		EnableInterrupts(state)
		panic("sync: condition wait from interrupt context")
	}

	g := currentGoroutine
	if g == nil {
		EnableInterrupts(state)
		panic("sync: condition wait outside a goroutine")
	}

	if g.waitAddr != nil || g.waitNext != nil {
		EnableInterrupts(state)
		panic("sync: goroutine is already queued on a runtime wait list")
	}

	g.waitAddr = unsafe.Pointer(addr)
	enqueueMutexWaiter(mutexRootFor(addr), g)
	return state
}

// condPark parks the current goroutine previously registered by condEnqueue
// and blocks until condWake dequeues it. Interrupts must still be disabled
// from the condEnqueue call; state is the value condEnqueue returned.
//
//go:export condPark runtime.condPark
func condPark(addr *uint32, state uint32) {
	g := currentGoroutine
	g.state = goroutineParked

	// Do not restore the caller's interrupt state while parked: it may have
	// interrupts masked, which would prevent PendSV and any waker from ever
	// running. The caller's state is restored after the wakeup.
	EnableInterrupts(0)

	for {
		for g.state == goroutineParked {
			gosched()
		}

		st := DisableInterrupts()
		if g.waitAddr == nil {
			// condWake dequeued this goroutine: a genuine wakeup.
			EnableInterrupts(st)
			break
		}

		// Spurious wakeup while still queued: park again.
		g.state = goroutineParked
		EnableInterrupts(st)
	}

	EnableInterrupts(state)
}

// condWake makes one (all=false) or every (all=true) goroutine waiting on
// addr runnable. It never blocks, so it is safe to call from interrupt
// handlers.
//
//go:export condWake runtime.condWake
func condWake(addr *uint32, all bool) {
	state := DisableInterrupts()

	root := mutexRootFor(addr)
	needsSchedule := false

	for {
		g := dequeueMutexWaiter(root, unsafe.Pointer(addr))
		if g == nil {
			break
		}

		if goreadyLocked(unsafe.Pointer(g)) {
			needsSchedule = true
		}

		if !all {
			break
		}
	}

	// Pend a single reschedule for the whole batch.
	if needsSchedule {
		gosched()
	}

	EnableInterrupts(state)
}
