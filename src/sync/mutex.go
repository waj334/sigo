package sync

import "sync/atomic"

type Mutex struct {
	state uint32
}

//sigo:extern gosched runtime.gosched
func gosched()

func (m *Mutex) Lock() {
	for !atomic.CompareAndSwapUint32(&m.state, 0, 1) {
		// Yield to run a different goroutine.
		gosched()
	}
}

func (m *Mutex) TryLock() bool {
	return atomic.CompareAndSwapUint32(&m.state, 0, 1)
}

func (m *Mutex) Unlock() {
	atomic.StoreUint32(&m.state, 0)
}
