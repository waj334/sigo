package sync

import "sync/atomic"

type Once struct {
	done atomic.Bool
	mu   Mutex
}

func (o *Once) Do(f func()) {
	if o.done.Load() {
		return
	}

	o.mu.Lock()

	// NOTE: The usage of defer here will unlock the mutex in the event f() panics.
	defer o.mu.Unlock()

	if !o.done.Load() {
		// NOTE: The usage of defer here will update the state in the event f() panics.
		defer o.done.Store(true)
		f()
	}
}
