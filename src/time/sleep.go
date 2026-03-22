package time

import (
	"sync"
	"unsafe"
)

//sigo:extern gopark runtime.gopark
//sigo:extern goparkWithCallback runtime.goparkWithCallback
//sigo:extern goresume runtime.goresume
//sigo:extern getg runtime.getg

func gopark(unsafe.Pointer)
func goparkWithCallback(unsafe.Pointer, func())
func goresume(unsafe.Pointer)
func getg() unsafe.Pointer

type sleepEntry struct {
	g        unsafe.Pointer
	deadline uint64
	next     *sleepEntry
}

var sleepQueue *sleepEntry
var sleepQueueMutex sync.Mutex

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

	// Insert into the linked list sorted by deadline.
	sleepQueueMutex.Lock()
	if sleepQueue == nil || deadline < sleepQueue.deadline {
		// Insert at head (either empty list or earliest deadline)
		entry.next = sleepQueue
		sleepQueue = entry
	} else {
		// Insert after an existing entry
		curr := sleepQueue
		for curr.next != nil && curr.next.deadline <= deadline {
			curr = curr.next
		}
		entry.next = curr.next
		curr.next = entry
	}
	sleepQueueMutex.Unlock()

	// CRITICAL: Use goparkWithCallback to arm timer atomically with parking.
	// This ensures the timer is armed only after the goroutine is marked as parked,
	// preventing race where timer fires before goroutine is parked.
	//
	// The sequence (all atomic):
	//   1. Disable interrupts (inside goparkWithCallback)
	//   2. Mark goroutine as parked
	//   3. Arm timer (in callback)
	//   4. Enable interrupts
	//   5. Yield
	goparkWithCallback(g, func() {
		addsleep(deadline)
	})
}

//go:export wake runtime.wake
func wake(t uint64) {
	sleepQueueMutex.Lock()

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

			// Unlock before resuming (goresume may trigger scheduling)
			sleepQueueMutex.Unlock()
			goresume(g)
			sleepQueueMutex.Lock()

			// Continue from the next entry (prev stays the same)
			curr = next
		} else {
			// Entry not ready yet, move to next
			prev = curr
			curr = curr.next
		}
	}

	sleepQueueMutex.Unlock()
}

func Sleep(d Duration) {
	if d > 0 {
		sleep(uint64(d))
	}
}
