package sync

import (
	"unsafe"
)

//sigo:extern waitGoroutine runtime.waitGoroutine
//sigo:extern resumeGoroutine runtime.resumeGoroutine
//sigo:extern runningGoroutine runtime.runningGoroutine

func waitGoroutine(unsafe.Pointer)
func resumeGoroutine(unsafe.Pointer)
func runningGoroutine() unsafe.Pointer

type Cond struct {
	L       Locker
	mutex   Mutex
	waiters []unsafe.Pointer
}

func NewCond(l Locker) *Cond {
	return &Cond{L: l}
}

func (c *Cond) Broadcast() {
	c.mutex.Lock()

	// Resume all waiting goroutines.
	for _, waiter := range c.waiters {
		resumeGoroutine(waiter)
	}

	// Clear the waiters list
	c.waiters = nil
	c.mutex.Unlock()
}

func (c *Cond) Signal() {
	c.mutex.Lock()

	if len(c.waiters) > 0 {
		// Pop the first waiter from the waiters list.
		waiter := c.waiters[0]
		if len(c.waiters) > 1 {
			c.waiters = c.waiters[1:]
		} else {
			// Clear the waiters list.
			c.waiters = nil
		}

		// Resume this goroutine.
		resumeGoroutine(waiter)
	}

	c.mutex.Unlock()
}

func (c *Cond) Wait() {
	// Add the current goroutine to the waiter list.
	c.mutex.Lock()
	c.waiters = append(c.waiters, runningGoroutine())
	c.mutex.Unlock()

	// Switch the current goroutine to the waiting state.
	c.L.Unlock()
	waitGoroutine(runningGoroutine())
	c.L.Lock()
}
