package time

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

type timerEntry struct {
	period   uint64
	nextTick uint64
	c        chan Time
	stopped  bool
	next     *timerEntry
}

// Ticker holds a channel that delivers "ticks" of a clock at intervals.
type Ticker struct {
	C     <-chan Time
	c     chan Time
	entry *timerEntry
}

var (
	timerQueue         *timerEntry
	timerQueueMutex    sync.Mutex
	timerGStarted      uint32
	timerGPtr          unsafe.Pointer
	timerWakeRequested uint32
)

// NewTicker returns a new Ticker containing a channel that will send
// the current time on the channel after each tick. The period of the
// ticks is specified by the duration argument. The ticker will adjust
// the time interval or drop ticks to make up for slow receivers.
// The duration d must be greater than zero; if not, NewTicker will panic.
// Stop the ticker to release associated resources.
func NewTicker(d Duration) *Ticker {
	if d <= 0 {
		panic("non-positive interval for NewTicker")
	}

	c := make(chan Time, 1)
	now := nanotime()
	entry := &timerEntry{
		period:   uint64(d),
		nextTick: now + uint64(d),
		c:        c,
	}

	timerQueueMutex.Lock()
	insertTimerEntry(entry)
	timerQueueMutex.Unlock()

	ensureTimerGoroutine()
	wakeTimerGoroutine()

	return &Ticker{
		C:     c,
		c:     c,
		entry: entry,
	}
}

// Stop turns off a ticker. After Stop, no more ticks will be sent.
// Stop does not close the channel, to prevent a concurrent goroutine
// reading from the channel from seeing an erroneous "tick".
func (t *Ticker) Stop() {
	timerQueueMutex.Lock()
	t.entry.stopped = true
	timerQueueMutex.Unlock()
	wakeTimerGoroutine()
}

// Reset stops a ticker and resets its period to the specified duration.
// The next tick will arrive after the new period elapses.
// The duration d must be greater than zero; if not, Reset will panic.
func (t *Ticker) Reset(d Duration) {
	if d <= 0 {
		panic("non-positive interval for Ticker.Reset")
	}

	timerQueueMutex.Lock()
	removeTimerEntry(t.entry)
	t.entry.period = uint64(d)
	t.entry.nextTick = nanotime() + uint64(d)
	t.entry.stopped = false
	insertTimerEntry(t.entry)
	timerQueueMutex.Unlock()

	wakeTimerGoroutine()
}

// Tick is a convenience wrapper for NewTicker providing access to the ticking
// channel only. While Tick is useful for clients that have no need to shut down
// the Ticker, be aware that without a way to shut it down the underlying
// Ticker cannot be recovered by the garbage collector; it "leaks".
func Tick(d Duration) <-chan Time {
	if d <= 0 {
		return nil
	}
	return NewTicker(d).C
}

// insertTimerEntry inserts an entry into timerQueue sorted by nextTick.
// Must be called with timerQueueMutex held.
func insertTimerEntry(entry *timerEntry) {
	// Guard: if the entry is already the head, don't re-insert.
	if timerQueue == entry {
		return
	}
	if timerQueue == nil || entry.nextTick < timerQueue.nextTick {
		entry.next = timerQueue
		timerQueue = entry
	} else {
		curr := timerQueue
		for curr.next != nil && curr.next.nextTick <= entry.nextTick {
			// Guard: if the entry is already present later in the queue,
			// don't create a cycle. This can happen when timerLoop drops
			// the mutex for a channel send and Reset re-inserts the same
			// entry before timerLoop re-acquires the mutex to reinsert it.
			if curr.next == entry {
				return
			}
			curr = curr.next
		}
		entry.next = curr.next
		curr.next = entry
	}
}

// removeTimerEntry removes a specific entry from timerQueue.
// Must be called with timerQueueMutex held.
func removeTimerEntry(target *timerEntry) {
	var prev *timerEntry
	for curr := timerQueue; curr != nil; curr = curr.next {
		if curr == target {
			if prev == nil {
				timerQueue = curr.next
			} else {
				prev.next = curr.next
			}
			target.next = nil
			return
		}
		prev = curr
	}
}

func ensureTimerGoroutine() {
	if atomic.CompareAndSwapUint32(&timerGStarted, 0, 1) {
		go timerLoop()
	}
}

func wakeTimerGoroutine() {
	atomic.StoreUint32(&timerWakeRequested, 1)
	g := timerGPtr
	if g != nil {
		goresume(g)
	}
}

func timerLoop() {
	timerGPtr = getg()

	for {
		// Check for pending wake request before doing any work.
		atomic.CompareAndSwapUint32(&timerWakeRequested, 1, 0)

		now := nanotime()
		var nearest uint64 = ^uint64(0)

		timerQueueMutex.Lock()

		var prev *timerEntry
		curr := timerQueue
		for curr != nil {
			next := curr.next

			if curr.stopped {
				// Remove stopped entry from the list.
				if prev == nil {
					timerQueue = next
				} else {
					prev.next = next
				}
				curr.next = nil
				curr = next
				continue
			}

			if now >= curr.nextTick {
				// Timer is due. Remove from current position.
				if prev == nil {
					timerQueue = next
				} else {
					prev.next = next
				}
				curr.next = nil

				// Advance the deadline by the period.
				curr.nextTick += curr.period
				// If we fell behind, skip forward to the next future tick.
				if curr.nextTick <= now {
					missed := (now - curr.nextTick) / curr.period
					curr.nextTick += (missed + 1) * curr.period
				}

				// Unlock before channel send — the send may interact with
				// other goroutines that could call Stop/Reset.
				timerQueueMutex.Unlock()

				// Non-blocking send: drop the tick if the receiver hasn't
				// consumed the previous one (standard Go Ticker behavior).
				select {
				case curr.c <- Time{t: now}:
				default:
				}

				// Re-insert with the updated deadline.
				timerQueueMutex.Lock()
				insertTimerEntry(curr)

				if curr.nextTick < nearest {
					nearest = curr.nextTick
				}

				// Restart scan from head since the list was modified.
				prev = nil
				curr = timerQueue
				continue
			}

			// Not due yet — track the nearest deadline.
			if curr.nextTick < nearest {
				nearest = curr.nextTick
			}
			prev = curr
			curr = next
		}

		timerQueueMutex.Unlock()

		// Check if a wake was requested while we were processing.
		if atomic.CompareAndSwapUint32(&timerWakeRequested, 1, 0) {
			continue
		}

		if nearest == ^uint64(0) {
			// No active timers. Park until a new ticker is added.
			gopark(getg())
		} else {
			now = nanotime()
			if nearest > now {
				sleep(nearest - now)
			}
		}
	}
}
