package runtime

import (
	"unsafe"
)

// coro is a stackful symmetric coroutine. It bypasses the goroutine
// scheduler — coroswitch swaps SP synchronously between the parent and the
// coro. The coro is logically a deep function call from its parent's
// perspective; it does NOT participate in scheduling.
//
// Layout note: sp and stack must be the first two fields. The chip-support
// initCoro reaches them via a mirrored *_coro struct that only includes
// these two fields.
type coro struct {
	sp    unsafe.Pointer // saved SP of whichever side is parked
	stack unsafe.Pointer // base of the coro's stack
	fn    func(*coro)    // user-supplied function (closure value)
	val   unsafe.Pointer // optional value passed across switches
	done  bool           // set by coroEntry when the user function returns
}

// Coro is a type alias making the coro type accessible to other packages
// (notably the iter package) without breaking the lowercase convention of
// the runtime API. The iter package uses this so its `*coro` matches
// runtime's `*coro` at the type-system level for the linkname'd
// runtime.newcoro / runtime.coroswitch calls.
type Coro = coro

//sigo:extern coroStackSize runtime._coroStackSize
var coroStackSize uintptr

// initCoro is provided by the chip-support package. It lays out the coro's
// initial stack frame so the first coroswitch's pop+bx lands in coroEntry.
//
//sigo:extern initCoro runtime.initCoro
func initCoro(c unsafe.Pointer)

// coroswitch_inner is the per-arch register-save/restore primitive. Saves
// callee-saved registers + LR on the current stack, swaps SP with c.sp,
// pops callee-saved + LR from the new stack, and returns.
//
//sigo:extern coroswitch_inner runtime.coroswitch_inner
func coroswitch_inner(c unsafe.Pointer)

//go:export newcoro runtime.newcoro
func newcoro(f func(*coro)) *coro {
	fv := *(*_func)(unsafe.Pointer(&f))
	stackSize := fv.stackSize
	if stackSize == 0 {
		stackSize = coroStackSize
	}

	state := DisableInterrupts()

	c := &coro{
		fn: f,
	}
	c.stack = alloc(stackSize)
	c.sp = c.stack // initCoro adjusts to the top of the initial frame
	initCoro(unsafe.Pointer(c))

	EnableInterrupts(state)
	return c
}

// coroswitch performs a symmetric switch between the parent and the coro.
// First call after newcoro: enters the coro and runs its user function.
// Subsequent calls: alternates between parent and coro.
//
//go:export coroswitch runtime.coroswitch
func coroswitch(c *coro) {
	if c.done {
		panic("coroswitch on a coro that has already completed")
	}
	state := DisableInterrupts()
	coroswitch_inner(unsafe.Pointer(c))
	EnableInterrupts(state)
}

// coroEntry is the trampoline that the coro's initial stack frame lands in
// on first switch. It re-enables interrupts (the parent disabled them
// during the switch), runs the user function, then performs a final switch
// back to the parent. The parent observes c.done = true and any subsequent
// coroswitch on this coro panics.
//
// panicYieldAfterStop is called by range-over-func closures when the
// iterator violates the protocol by calling yield after yield previously
// returned false. Per Go 1.23 spec, this is a runtime panic.
//
//go:export panicYieldAfterStop runtime.panicYieldAfterStop
func panicYieldAfterStop() {
	panic("range-over-func: yield called after iterator stopped")
}

//sigo:export coroEntry runtime.coroEntry
//sigo:attribute coroEntry noreturn
func coroEntry(c *coro) {
	// The parent's coroswitch disabled interrupts before the asm switch.
	// Re-enable them now that we're safely on the coro stack and the swap
	// is complete. The user function runs with interrupts enabled.
	EnableInterrupts(0)

	c.fn(c)

	// User function returned. Mark done and switch back. The public
	// coroswitch wrapper would panic on c.done == true, so call the inner
	// asm directly. Disable interrupts around the swap as on entry.
	c.done = true
	DisableInterrupts()
	coroswitch_inner(unsafe.Pointer(c))

	// If we ever resume here (parent called coroswitch despite done == true,
	// which would have panicked at the wrapper), spin. This is unreachable
	// in well-formed code.
	for {
	}
}
