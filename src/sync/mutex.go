package sync

import "sync/atomic"

type Mutex struct {
	state int32
}

//go:linkname runScheduler runtime.runScheduler
func runScheduler() bool

func (m *Mutex) Lock() {
	for !atomic.CompareAndSwapInt32(&m.state, 0, 1) {
		// Yield
		runScheduler()
	}
}

func (m *Mutex) TryLock() bool {
	return atomic.CompareAndSwapUint32(&m.state, 0, 1)
}

func (m *Mutex) Unlock() {
	atomic.StoreInt32(&m.state, 0)
}
