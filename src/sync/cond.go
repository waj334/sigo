package sync

import (
	"unsafe"
)

//sigo:extern gopark runtime.gopark
//sigo:extern goresume runtime.goresume
//sigo:extern getg runtime.getgPtr

func gopark(unsafe.Pointer)
func goresume(unsafe.Pointer)
func getg() unsafe.Pointer

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
		goresume(waiter)
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
		goresume(waiter)
	}

	c.mutex.Unlock()
}

func (c *Cond) Wait() {
	// Add the current goroutine to the waiter list.
	c.mutex.Lock()
	c.waiters = append(c.waiters, getg())
	c.mutex.Unlock()

	// Switch the current goroutine to the waiting state.
	c.L.Unlock()
	gopark(getg())
	c.L.Lock()
}
