package sync

//sigo:extern condEnqueue runtime.condEnqueue
//sigo:extern condPark runtime.condPark
//sigo:extern condWake runtime.condWake

func condEnqueue(addr *uint32) uint32
func condPark(addr *uint32, state uint32)
func condWake(addr *uint32, all bool)

// Cond implements a condition variable, a rendezvous point for goroutines
// waiting for or announcing the occurrence of an event.
//
// Each Cond has an associated Locker L (often a *Mutex), which must be held
// when calling Wait.
//
// A Cond must not be copied after first use.
type Cond struct {
	_ noCopy

	// L is held while observing or changing the condition.
	L Locker

	// notify is never read or written directly; its address identifies this
	// Cond's wait queue inside the runtime.
	notify uint32
}

func NewCond(l Locker) *Cond {
	return &Cond{L: l}
}

// Broadcast wakes all goroutines waiting on c.
//
// It is allowed but not required for the caller to hold c.L during the call.
func (c *Cond) Broadcast() {
	condWake(&c.notify, true)
}

// Signal wakes one goroutine waiting on c, if there is any.
//
// It is allowed but not required for the caller to hold c.L during the call.
func (c *Cond) Signal() {
	condWake(&c.notify, false)
}

// Wait atomically unlocks c.L and suspends execution of the calling
// goroutine. After later resuming execution, Wait locks c.L before returning.
//
// Because c.L is not locked while Wait is waiting, the caller typically
// cannot assume that the condition is true when Wait returns. Instead, the
// caller should Wait in a loop:
//
//	c.L.Lock()
//	for !condition() {
//	    c.Wait()
//	}
//	... make use of condition ...
//	c.L.Unlock()
func (c *Cond) Wait() {
	// condEnqueue returns with interrupts disabled, so no Signal or Broadcast
	// can run between queueing, releasing L, and parking. This is what makes
	// the unlock-then-park sequence free of lost wakeups.
	state := condEnqueue(&c.notify)
	c.L.Unlock()
	condPark(&c.notify, state)
	c.L.Lock()
}
