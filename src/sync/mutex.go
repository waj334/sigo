package sync

import isync "internal/sync"

// A Mutex is a mutual exclusion lock.
//
// The zero value for a Mutex is an unlocked mutex.
//
// A Mutex must not be copied after first use.
//
// In the terminology of the Go memory model, the n'th call to Mutex.Unlock
// “synchronizes before” the m'th call to Mutex.Lock for any n < m.
// A successful call to Mutex.TryLock is equivalent to a call to Lock.
// A failed call to TryLock does not establish any “synchronizes before”
// relation at all.
type Mutex struct {
	_  noCopy
	mu isync.Mutex
}

// Lock locks m.
// If the lock is already in use, the calling goroutine blocks until the mutex
// is available.
func (m *Mutex) Lock() {
	m.mu.Lock()
}

// TryLock tries to lock m and reports whether it succeeded.
//
// Note that while correct uses of TryLock do exist, they are rare, and use of
// TryLock is often a sign of a deeper problem in a particular use of mutexes.
func (m *Mutex) TryLock() bool {
	return m.mu.TryLock()
}

// Unlock unlocks m.
// It is a run-time error if m is not locked on entry to Unlock.
//
// A locked Mutex is not associated with a particular goroutine. It is allowed
// for one goroutine to lock a Mutex and then arrange for another goroutine to
// unlock it.
func (m *Mutex) Unlock() {
	m.mu.Unlock()
}
