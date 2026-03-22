package sync

import (
	"sync/atomic"
)

//sigo:extern gosched runtime.gosched
func gosched()

type Mutex struct {
	locked uint32
}

func (m *Mutex) Lock() {
	for !atomic.CompareAndSwapUint32(&m.locked, 0, 1) {
		gosched()
	}
}

// TryLock attempts to acquire the lock without blocking.
// Returns true if the lock was acquired, false otherwise.
func (m *Mutex) TryLock() bool {
	return atomic.CompareAndSwapUint32(&m.locked, 0, 1)
}

func (m *Mutex) Unlock() {
	atomic.StoreUint32(&m.locked, 0)
}
