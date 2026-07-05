package runtime

import (
	"sync/atomic"
	"unsafe"
)

// mutexRootCount must be a power of two.
const mutexRootCount = 16

// mutexRoot holds parked goroutines whose mutex addresses hash to this root.
// A goroutine's waitAddr distinguishes queues that collide in the same root.
type mutexRoot struct {
	head *goroutine
	tail *goroutine
}

// mutexRoots is protected by DisableInterrupts on the current single-core
// runtime. A future multicore runtime must replace that protection with a
// cross-core lock or hardware semaphore.
var mutexRoots [mutexRootCount]mutexRoot

func mutexRootFor(addr *uint32) *mutexRoot {
	index := (uintptr(unsafe.Pointer(addr)) >> 2) & (mutexRootCount - 1)
	return &mutexRoots[index]
}

func enqueueMutexWaiter(root *mutexRoot, g *goroutine) {
	g.waitNext = nil

	if root.tail == nil {
		root.head = g
		root.tail = g
		return
	}

	root.tail.waitNext = g
	root.tail = g
}

func dequeueMutexWaiter(root *mutexRoot, addr unsafe.Pointer) *goroutine {
	var prev *goroutine
	curr := root.head

	for curr != nil {
		next := curr.waitNext
		if curr.waitAddr == addr {
			if prev == nil {
				root.head = next
			} else {
				prev.waitNext = next
			}

			if root.tail == curr {
				root.tail = prev
			}

			curr.waitNext = nil
			curr.waitAddr = nil
			return curr
		}

		prev = curr
		curr = next
	}

	return nil
}

// mutexLockSlow is the allocation-free contended path for internal/sync.Mutex.
//
// The public fast path has already observed the mutex as locked. This function
// rechecks while the scheduler state is protected, queues the current goroutine,
// marks it parked, and yields. Unlock transfers ownership directly to the
// oldest waiter, leaving the lock word set.
//
//go:export mutexLockSlow runtime.mutexLockSlow
func mutexLockSlow(addr *uint32) {
	interruptState := DisableInterrupts()

	// The mutex may have become free between the failed fast path and entering
	// this protected slow path.
	if atomic.CompareAndSwapUint32(addr, 0, 1) {
		EnableInterrupts(interruptState)
		return
	}

	// Blocking is impossible from an actual exception handler. Interrupts
	// merely being masked in Thread mode is not an error; that case is handled
	// by temporarily enabling them while this goroutine is parked.
	if InInterrupt() {
		EnableInterrupts(interruptState)
		panic("sync: blocking mutex lock from interrupt context")
	}

	g := currentGoroutine
	if g == nil {
		EnableInterrupts(interruptState)
		panic("sync: mutex lock outside a goroutine")
	}

	if g.waitAddr != nil || g.waitNext != nil {
		EnableInterrupts(interruptState)
		panic("sync: goroutine is already queued on a runtime wait list")
	}

	g.waitAddr = unsafe.Pointer(addr)
	enqueueMutexWaiter(mutexRootFor(addr), g)

	// Publish the parked state before allowing Unlock to run.
	g.state = goroutineParked

	// Do not restore interruptState here. It might indicate that interrupts
	// were disabled before Lock was called, which would prevent PendSV and the
	// lock owner from running. Temporarily enable interrupts while parked.
	EnableInterrupts(0)

	for {
		for g.state == goroutineParked {
			gosched()
		}

		state := DisableInterrupts()
		if g.waitAddr == nil {
			// mutexUnlock dequeued this goroutine before waking it, so this is
			// a genuine handoff: this goroutine now owns the lock.
			EnableInterrupts(state)
			break
		}

		// Spurious wakeup while still on the wait queue. Returning here would
		// claim a lock this goroutine does not own; park again instead.
		g.state = goroutineParked
		EnableInterrupts(state)
	}

	// Unlock handed ownership directly to this goroutine. Restore the
	// interrupt state that was active when Lock was called.
	EnableInterrupts(interruptState)
}

// mutexUnlock releases a sync.Mutex or transfers it to the oldest waiter.
//
//go:export mutexUnlock runtime.mutexUnlock
func mutexUnlock(addr *uint32) {
	state := DisableInterrupts()

	if atomic.LoadUint32(addr) == 0 {
		EnableInterrupts(state)
		panic("sync: unlock of unlocked mutex")
	}

	waiter := dequeueMutexWaiter(
		mutexRootFor(addr),
		unsafe.Pointer(addr),
	)

	if waiter == nil {
		atomic.StoreUint32(addr, 0)
		EnableInterrupts(state)
		return
	}

	// Direct handoff: addr remains locked. Only the selected waiter is made
	// runnable, and it returns from mutexLockSlow as the new owner.
	needsSchedule := goreadyLocked(unsafe.Pointer(waiter))
	if needsSchedule {
		gosched()
	}

	EnableInterrupts(state)
}

type mutex struct {
	state uint32
}

func (m *mutex) lock() {
	if atomic.CompareAndSwapUint32(&m.state, 0, 1) {
		return
	}

	mutexLockSlow(&m.state)
}

func (m *mutex) tryLock() bool {
	return atomic.CompareAndSwapUint32(&m.state, 0, 1)
}

func (m *mutex) unlock() {
	mutexUnlock(&m.state)
}
