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

	waitNext *goroutine
	waitAddr unsafe.Pointer
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
//go:export gopark runtime.gopark
//go:export goparkRestore runtime.goparkRestore
//go:export goready runtime.goready
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

// schedule selects the context PendSV should restore.
//
// PendSV has already saved currentGoroutine's complete software context and
// published its stackTop before calling this function. Interrupts remain
// disabled for the entire call, so the goroutine ring and scheduler state are
// stable while the decision is made.
func schedule() bool {
	// currentGoroutine still names the context that was physically active on
	// PSP at PendSV entry. Its context is now durable in outgoing.stackTop.
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
		// Consume the targeted wakeup exactly once. targetGoroutine is only a
		// fast-path hint; every ready goroutine remains discoverable by the ring
		// scan below.
		target := targetGoroutine
		targetGoroutine = nil

		if target == outgoing {
			// The goroutine was resumed after marking itself parked but before an
			// earlier pending PendSV switched away from it. Its saved context is
			// valid, but the physically active context is still this goroutine.
			if outgoing.state == goroutineReady {
				outgoing.state = goroutineRunning
			}

			if outgoing.state == goroutineRunning {
				lastGoroutine = outgoing
				return false
			}

			// If outgoing is parked or exiting, ignore this stale target and
			// continue with normal ring selection.
		} else if runnable(target) &&
			(outgoing.state == goroutineParked ||
				outgoing.state == goroutineExiting) {

			// Honor the direct-handoff hint only when the outgoing goroutine
			// is giving up the CPU. When outgoing still wants to run, fall
			// through to the ring scan so a continuous stream of targeted
			// wakeups cannot starve other ready goroutines.
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
		// No actual context change occurred. PendSV will return using the
		// hardware frame already present on PSP; the software save made before
		// this call is simply the latest durable copy of the same context.
		if outgoing != nil && outgoing.state == goroutineReady {
			outgoing.state = goroutineRunning
		}

		lastGoroutine = outgoing
		return false
	}

	// Commit the decision. PendSV already saved outgoing and will restore the
	// context named by currentGoroutine after this function returns.
	lastGoroutine = outgoing
	currentGoroutine = next

	if outgoing != nil {
		switch outgoing.state {
		case goroutineRunning:
			outgoing.state = goroutineReady

		case goroutineExiting:
			// The outgoing context was saved before schedule was called, so it is
			// now safe to unlink it. Reclamation is still deferred because this
			// code runs in PendSV exception context and must not invoke the normal
			// allocator or free the stack synchronously.
			unlinkGoroutine(outgoing)
			retireGoroutine(outgoing)
		}
	}

	next.state = goroutineRunning
	return true
}

func addGoroutine(f _func) {
	if f.f == nil {
		return
	}

	stackSize := f.stackSize
	if stackSize == 0 {
		stackSize = goroutineStackSize
	}

	// Allocate and initialize outside the interrupts-disabled region. The raw
	// stack allocation is serialized with every other allocator user through
	// gcMu: calling malloc with interrupts merely disabled cannot exclude a
	// goroutine that was preempted inside the allocator while holding gcMu.
	gcMu.lock()
	stack := malloc(stackSize)
	gcMu.unlock()
	if stack == nil {
		abort()
	}

	newGoroutine := &goroutine{
		stack: stack,
		// initGoroutine may move stackTop depending on the target's stack
		// growth direction.
		stackTop: stack,
		__func: _func{
			f:         f.f,
			args:      f.args,
			stackSize: stackSize,
		},
		state: goroutineNotStarted,
	}

	gint := uintptr(unsafe.Pointer(newGoroutine))
	newGoroutine.chacha8.Init64([4]uint64{
		uint64(gint >> 8),
		uint64(gint >> 16),
		uint64(gint >> 24),
		uint64(gint >> 32),
	})

	initGoroutine(unsafe.Pointer(newGoroutine))

	// Only the ring linkage itself must be atomic with respect to the
	// scheduler.
	state := DisableInterrupts()

	oldHead := headGoroutine
	headGoroutine = newGoroutine
	if oldHead == nil {
		headGoroutine.next = headGoroutine
		headGoroutine.prev = headGoroutine
	} else {
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
// executing on g's stack. PendSV saves that context first, then schedule
// unlinks it after selecting a different incoming goroutine.
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
	goparkRestore(ptr, state)
}

// goparkRestore parks a goroutine while interrupts are already disabled,
// restores the caller's previous interrupt state, and waits until the
// goroutine is resumed.
//
// The caller must not restore state independently after calling this function.
func goparkRestore(ptr unsafe.Pointer, state uint32) {
	g := (*goroutine)(ptr)
	g.state = goroutineParked

	EnableInterrupts(state)

	for g.state == goroutineParked {
		gosched()
	}
}

// goready transitions a parked goroutine into a runnable state. It does not
// pend PendSV; callers that make one or more goroutines ready should request a
// single schedule after completing the batch.
func goready(ptr unsafe.Pointer) bool {
	state := DisableInterrupts()
	ready := goreadyLocked(ptr)
	EnableInterrupts(state)
	return ready
}

// goreadyLocked requires interrupts to be disabled.
func goreadyLocked(ptr unsafe.Pointer) bool {
	g := (*goroutine)(ptr)
	if g == nil || g.state != goroutineParked {
		return false
	}

	// The wake happened after the goroutine marked itself parked but before
	// PendSV switched away from its physically active context.
	if g == currentGoroutine {
		g.state = goroutineRunning
		return false
	}

	g.state = goroutineReady

	// This is a scheduling hint only. Overwriting it is safe because all ready
	// goroutines remain discoverable by the normal ring scan.
	targetGoroutine = g
	return true
}

func goresume(ptr unsafe.Pointer) {
	if goready(ptr) {
		gosched()
	}
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
// callers in packages such as time and sync can hold an opaque goroutine
// handle without referencing runtime's private goroutine type.
//
//go:export getgPtr runtime.getgPtr
func getgPtr() unsafe.Pointer {
	return unsafe.Pointer(currentGoroutine)
}

// unlinkGoroutine removes g from the runnable ring.
//
// PendSV has already saved g's context before schedule calls this function.
// The stack is not freed here because schedule runs in exception context.
func unlinkGoroutine(g *goroutine) {
	// If the incremental GC root scan is positioned on this goroutine, abandon
	// the remainder of its stack scan and retarget the cursor so the scan
	// resumes with the ring successor. Without this, advanceScanState would
	// follow g.next after it has been repurposed for the retired list and scan
	// freed memory or dereference nil.
	if gc.phase == gcMark &&
		gc.scanState == gcScanGoroutines &&
		gc.currentGoroutine == g {

		if g.next == g {
			// The ring is about to become empty; move the scan directly to
			// the globals phase.
			gc.currentGoroutine = nil
			gc.scanState = gcScanGlobals
			gc.currentAddress = gcGlobalsStart()
			gc.endAddress = gcGlobalsEnd()
		} else {
			gc.currentGoroutine = g.prev
			gc.currentAddress = gc.endAddress
		}
	}

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
// free the stack there. The retired list is consumed later in Thread mode by
// reapRetiredGoroutines.
func retireGoroutine(g *goroutine) {
	g.next = retiredGoroutines
	retiredGoroutines = g
}

// reapRetiredGoroutines releases the resources of goroutines that schedule
// retired after they exited. It must run in Thread mode: schedule cannot free
// stacks from PendSV exception context.
//
// Without this consumer, exited goroutines and their stacks are orphaned
// forever and the heap is eventually exhausted.
func reapRetiredGoroutines() {
	state := DisableInterrupts()
	g := retiredGoroutines
	retiredGoroutines = nil
	EnableInterrupts(state)

	for g != nil {
		next := g.next
		g.next = nil

		if g.stack != nil {
			// Serialize with every other allocator user.
			gcMu.lock()
			free(g.stack)
			gcMu.unlock()

			g.stack = nil
			g.stackTop = nil
		}

		// Dropping all references lets the collector reclaim the goroutine
		// struct itself on a later cycle.
		g = next
	}
}
