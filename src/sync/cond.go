package sync

import (
	"unsafe"
)

//sigo:extern waitTask runtime.waitTask
//sigo:extern resumeTask runtime.resumeTask
//sigo:extern runningTask runtime.runningTask

func waitTask(unsafe.Pointer)
func resumeTask(unsafe.Pointer)
func runningTask() unsafe.Pointer

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

	// Resume all waiting goroutines
	for _, waiter := range c.waiters {
		resumeTask(waiter)
	}

	// Clear the waiters list
	c.waiters = nil
	c.mutex.Unlock()
}

func (c *Cond) Signal() {
	c.mutex.Lock()

	if len(c.waiters) > 0 {
		// Pop the first waiter from the waiters list
		waiter := c.waiters[0]
		if len(c.waiters) > 1 {
			c.waiters = c.waiters[1:]
		} else {
			// Clear the waiters list
			c.waiters = nil
		}

		// Resume this goroutine
		resumeTask(waiter)
	}

	c.mutex.Unlock()
}

func (c *Cond) Wait() {
	// Add the current task to the waiter list
	c.mutex.Lock()
	c.waiters = append(c.waiters, runningTask())
	c.mutex.Unlock()

	// Switch the current task to the waiting state
	c.L.Unlock()
	waitTask(runningTask())
	c.L.Lock()
}
