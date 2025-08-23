package sync

import (
	"sync/atomic"
	"unsafe"
)

type Mutex struct {
	next   uint32
	owner  uint32
	ownerg unsafe.Pointer
}

//sigo:extern gosched runtime.gosched
func gosched()

func (m *Mutex) Lock() {
	currg := getg()
	ownerg := atomic.LoadPointer(&m.ownerg)

	// Detect recursive locking (non-reentrant).
	if (currg == nil && ownerg != nil) || (currg != nil && ownerg == currg) {
		panic("sync.Mutex: double lock by same goroutine")
	}

	// Get a ticket for the current goroutine.
	ticket := atomic.AddUint32(&m.next, 1) - 1

	// Wait until it's this goroutine's turn to acquire the lock.
	for atomic.LoadUint32(&m.owner) != ticket {
		// Yield to run a different goroutine.
		gosched()
	}

	// The current goroutine will own the lock.
	atomic.StorePointer(&m.ownerg, currg)
}

func (m *Mutex) TryLock() bool {
	owner := atomic.LoadUint32(&m.owner)
	if atomic.LoadUint32(&m.next) != owner {
		// The lock is already acquired.
		return false
	}

	// Attempt to claim the next ticket.
	if !atomic.CompareAndSwapUint32(&m.next, owner, owner+1) {
		return false
	}

	// The current goroutine will own the lock.
	atomic.StorePointer(&m.ownerg, getg())
	return true
}

func (m *Mutex) Unlock() {
	if atomic.LoadUint32(&m.owner) == atomic.LoadUint32(&m.next) {
		// No one holds the lock (no outstanding ticket == owner).
		panic("sync.Mutex: unlock of unlocked mutex")
	}

	// The lock can only be unlocked by the original owner. Check this first!
	if atomic.LoadPointer(&m.ownerg) != getg() {
		panic("sync.Mutex: unlock by non-owner")
	}

	// Free the lock from the current owner.
	atomic.StorePointer(&m.ownerg, nil)

	// Hand off to the next ticket holder (FIFO).
	atomic.AddUint32(&m.owner, 1)
}
