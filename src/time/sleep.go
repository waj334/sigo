package time

import (
	"unsafe"
)

//sigo:extern gopark runtime.gopark
//sigo:extern goparkWithCallback runtime.goparkWithCallback
//sigo:extern goresume runtime.goresume
//sigo:extern getg runtime.getg
//sigo:extern DisableInterrupts runtime.DisableInterrupts
//sigo:extern EnableInterrupts runtime.EnableInterrupts

func gopark(unsafe.Pointer)
func goparkWithCallback(unsafe.Pointer, func())
func goresume(unsafe.Pointer)
func getg() unsafe.Pointer
func DisableInterrupts() uint32
func EnableInterrupts(uint32)

type sleepEntry struct {
	g        unsafe.Pointer
	deadline uint64
	next     *sleepEntry
}

var sleepQueue *sleepEntry

//sigo:export addsleep runtime.addsleep
//sigo:linkage addsleep weak
func addsleep(uint64) {
	// By default, this does nothing. When implemented, this function can be the entry point for setting up hardware
	// timing mechanisms.
}

func sleep(d uint64) {
	g := getg()
	if g == nil {
		panic("sleep called from non-goroutine")
	}

	deadline := nanotime() + d

	// Create a new sleep entry.
	entry := &sleepEntry{
		g:        g,
		deadline: deadline,
	}

	// CRITICAL: Insert into the sleep queue AND arm the timer atomically
	// with parking the goroutine. All three operations happen inside
	// goparkWithCallback with interrupts disabled. This prevents:
	//   - wake() from removing the entry before the goroutine is parked
	//     (goresume would be a no-op since goroutine isn't parked yet,
	//     causing the goroutine to sleep forever)
	//   - The timer firing before the goroutine is parked
	goparkWithCallback(g, func() {
		// Interrupts are disabled here (goparkWithCallback holds them).
		if sleepQueue == nil || deadline < sleepQueue.deadline {
			entry.next = sleepQueue
			sleepQueue = entry
		} else {
			curr := sleepQueue
			for curr.next != nil && curr.next.deadline <= deadline {
				curr = curr.next
			}
			entry.next = curr.next
			curr.next = entry
		}
		addsleep(deadline)
	})
}

//go:export wake runtime.wake
func wake(t uint64) {
	// t must be in nanoseconds (same units as nanotime()).
	// The platform's alarm callback is responsible for converting hardware
	// tick counts to nanoseconds before calling wake. Safe to call from
	// interrupt context: uses DisableInterrupts/EnableInterrupts instead of
	// a spinlock mutex, and goresume() is ISR-safe (sets a flag and pends PendSV).
	state := DisableInterrupts()

	// Wake all goroutines whose deadlines have passed
	var prev *sleepEntry
	curr := sleepQueue

	for curr != nil {
		if t >= curr.deadline {
			g := curr.g
			next := curr.next

			// Remove from sleep queue
			if prev == nil {
				sleepQueue = next
			} else {
				prev.next = next
			}

			// Re-enable interrupts around goresume: it sets targetGoroutine
			// and pends PendSV, both safe from ISR. Re-disabling afterward
			// protects the next queue traversal step.
			EnableInterrupts(state)
			goresume(g)
			state = DisableInterrupts()

			// Continue from the next entry (prev stays the same)
			curr = next
		} else {
			// Entry not yet due, move to next
			prev = curr
			curr = curr.next
		}
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

//sigo:export nextsleep runtime.nextsleep
//sigo:linkage nextsleep weak
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
	if d > 0 {
		sleep(uint64(d))
	}
}
