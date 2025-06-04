package time

import (
	"sync"
	"unsafe"
)

//sigo:extern gopark runtime.gopark
//sigo:extern goresume runtime.goresume
//sigo:extern getg runtime.getg

func gopark(unsafe.Pointer)
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

	// Insert into the linked list.
	sleepQueueMutex.Lock()
	if sleepQueue == nil {
		sleepQueue = entry
	} else {
		// Insert into the list at the appropriate location sorted by deadline.
		curr := sleepQueue
		for curr.next != nil && curr.next.deadline <= entry.deadline {
			curr = curr.next
		}
		entry.next = curr.next
		curr.next = entry
	}

	addsleep(deadline)
	sleepQueueMutex.Unlock()

	// Schedule another goroutine to begin running.
	gopark(g)
}

//go:export wake runtime.wake
func wake(t uint64) {
	entry := sleepQueue
	var last *sleepEntry

	for entry != nil {
		if t > entry.deadline {
			g := entry.g

			// Remove from sleep queue.
			sleepQueueMutex.Lock()
			if last != nil {
				last.next = entry.next
			} else {
				sleepQueue = entry.next
			}
			sleepQueueMutex.Unlock()

			// Advance to the next entry.
			entry = entry.next

			// Resume this goroutine.
			goresume(g)
			continue
		}

		// Advance last and entry if not removed.
		last = entry
		entry = entry.next
	}
}

func Sleep(d Duration) {
	if d > 0 {
		sleep(uint64(d))
	}
}
