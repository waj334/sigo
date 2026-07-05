package time

import (
	"runtime"
	"unsafe"
)

type sleepEntry struct {
	g        unsafe.Pointer
	deadline uint64
	next     *sleepEntry
}

var sleepQueue *sleepEntry

func sleep(d uint64) {
	// Allocate before disabling interrupts: allocation may park on the
	// allocator lock, which briefly re-enables interrupts.
	entry := &sleepEntry{}

	state := DisableInterrupts()
	sleepOn(entry, d, state)
}

// sleepOn queues the current goroutine on the sleep queue using the provided
// (unqueued) entry and parks until the deadline passes or the goroutine is
// resumed directly. Interrupts must already be disabled; state is the
// caller's saved interrupt state, restored per goparkRestore's protocol.
//
// On return the entry is guaranteed to no longer be on the sleep queue, so a
// caller may reuse it and a stale entry can never fire a wakeup at a
// goroutine that has moved on to wait on something else.
func sleepOn(entry *sleepEntry, d uint64, state uint32) {
	g := getg()
	if g == nil {
		EnableInterrupts(state)
		panic("sleep called from non-goroutine")
	}

	entry.g = g
	entry.deadline = nanotime() + d
	entry.next = nil

	arm := false

	if sleepQueue == nil || entry.deadline < sleepQueue.deadline {
		entry.next = sleepQueue
		sleepQueue = entry
		arm = true
	} else {
		curr := sleepQueue
		for curr.next != nil &&
			curr.next.deadline <= entry.deadline {

			curr = curr.next
		}

		entry.next = curr.next
		curr.next = entry
	}

	if arm {
		addsleep(entry.deadline)
	}

	goparkRestore(g, state)

	// The wakeup may have come from goresume rather than the timer. Remove
	// the entry if it is still queued (a no-op when wake already removed it)
	// so it cannot fire later at a goroutine that is parked on something else.
	cleanup := DisableInterrupts()
	removeSleepEntry(entry)
	EnableInterrupts(cleanup)
}

//go:export wake runtime.wake
func wake(t uint64) {
	state := DisableInterrupts()

	var ready *sleepEntry

	for sleepQueue != nil && sleepQueue.deadline <= t {
		entry := sleepQueue
		sleepQueue = entry.next

		entry.next = ready
		ready = entry
	}

	if sleepQueue != nil {
		addsleep(sleepQueue.deadline)
	}

	needsSchedule := false

	for ready != nil {
		entry := ready
		ready = entry.next
		entry.next = nil

		if goready(entry.g) {
			needsSchedule = true
		}
	}

	// Pend PendSV while the entire wake operation is still atomic.
	if needsSchedule {
		runtime.Gosched()
	}

	EnableInterrupts(state)
}

// removeSleepEntry removes a specific entry from the sleepQueue.
// Must be called with interrupts disabled.
func removeSleepEntry(target *sleepEntry) {
	var prev *sleepEntry
	for curr := sleepQueue; curr != nil; curr = curr.next {
		if curr == target {
			if prev == nil {
				sleepQueue = curr.next
			} else {
				prev.next = curr.next
			}
			return
		}
		prev = curr
	}
}

//sigo:export runtime.nextsleep
//sigo:linkage weak
func nextsleep() uint64 {
	// Returns the deadline (in nanoseconds) of the next pending sleeper, or 0
	// if none. Called by platform code from interrupt context after wake()
	// returns, so it must not acquire any mutex. sleepQueue is stable during
	// ISR execution because no goroutine can run to modify it.
	if sleepQueue != nil {
		return sleepQueue.deadline
	}
	return 0
}

func Sleep(d Duration) {
	if d <= 0 {
		return
	}

	// sleep may return early if this goroutine is resumed directly, so keep
	// sleeping until the deadline has actually passed.
	deadline := nanotime() + uint64(d)
	for {
		now := nanotime()
		if now >= deadline {
			return
		}

		sleep(deadline - now)
	}
}

var fallbackSleepDeadline uint64

// addsleep arms the generic scheduler-tick fallback.
//
// This function is called with interrupts disabled. It must not block,
// allocate, or enable interrupts.
//
//sigo:export runtime.addsleep
//sigo:linkage weak
func addsleep(deadline uint64) {
	fallbackSleepDeadline = deadline
}

// checksleep services the weak scheduler-tick sleep timer.
//
// A strong platform addsleep implementation never writes
// fallbackSleepDeadline, so this becomes a cheap zero check.
//
//sigo:export runtime.checksleep
func checksleep(now uint64) {
	deadline := fallbackSleepDeadline
	if deadline == 0 || now < deadline {
		return
	}

	// Clear before wake. wake may arm the next queue head by calling
	// addsleep again, and that new value must not be overwritten here.
	fallbackSleepDeadline = 0
	wake(now)
}
