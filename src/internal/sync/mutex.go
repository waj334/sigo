package sync

import "sync/atomic"

const mutexLocked uint32 = 1

//sigo:extern runtime.mutexLockSlow
func runtimeMutexLockSlow(*uint32)

//sigo:extern runtime.mutexUnlock
func runtimeMutexUnlock(*uint32)

// Mutex is the internal implementation used by the public sync package.
//
// The zero value is an unlocked mutex. Mutex must not be copied after first
// use.
type Mutex struct {
	state uint32
}

// Lock locks m.
//
// The uncontended path is handled here. Contended locking is delegated to the
// runtime, which queues and parks the current goroutine without allocating.
func (m *Mutex) Lock() {
	if atomic.CompareAndSwapUint32(&m.state, 0, mutexLocked) {
		return
	}

	runtimeMutexLockSlow(&m.state)
}

// TryLock tries to lock m and reports whether it succeeded.
func (m *Mutex) TryLock() bool {
	return atomic.CompareAndSwapUint32(&m.state, 0, mutexLocked)
}

// Unlock unlocks m.
//
// The runtime handles both the uncontended release and direct handoff to a
// parked waiter. It panics if m is not locked.
func (m *Mutex) Unlock() {
	runtimeMutexUnlock(&m.state)
}
