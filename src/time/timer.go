package time

import "sync/atomic"

// Timer fires once after a duration. The current time is delivered on
// C when the timer expires. A Timer must be created with NewTimer or
// AfterFunc; the zero value is not usable.
type Timer struct {
	C       <-chan Time
	c       chan Time
	stopped uint32 // accessed atomically; 1 = canceled
}

func runTimer(t *Timer, dur uint64) {
	if dur > 0 {
		sleep(dur)
	}
	if atomic.LoadUint32(&t.stopped) == 0 {
		select {
		case t.c <- Time{t: nanotime()}:
		default:
		}
	}
}

func runAfterFunc(t *Timer, f func()) {
	if _, ok := <-t.C; ok {
		f()
	}
}

// NewTimer creates a new Timer that will send the current time on its
// channel after at least duration d.
func NewTimer(d Duration) *Timer {
	c := make(chan Time, 1)
	t := &Timer{C: c, c: c}
	var dur uint64
	if d > 0 {
		dur = uint64(d)
	}
	go runTimer(t, dur)
	return t
}

// After waits for the duration to elapse and then sends the current
// time on the returned channel. It is equivalent to NewTimer(d).C.
func After(d Duration) <-chan Time {
	return NewTimer(d).C
}

// AfterFunc waits for the duration to elapse and then calls f in its
// own goroutine. It returns a Timer that can be used to cancel the
// call using its Stop method.
func AfterFunc(d Duration, f func()) *Timer {
	t := NewTimer(d)
	go runAfterFunc(t, f)
	return t
}

// Stop prevents the Timer from firing. It returns true if the call
// stops the timer, false if the timer has already expired or been
// stopped. Stop does not close the channel, to prevent a read from
// the channel succeeding incorrectly.
func (t *Timer) Stop() bool {
	return atomic.SwapUint32(&t.stopped, 1) == 0
}

// Reset changes the timer to expire after duration d. It returns true
// if the timer had been active, false if the timer had expired or been
// stopped.
//
// For a Timer created with NewTimer, Reset should be invoked only on
// stopped or expired timers with drained channels (the standard Go
// idiom: if !t.Stop() { <-t.C }; t.Reset(d)).
func (t *Timer) Reset(d Duration) bool {
	wasActive := atomic.SwapUint32(&t.stopped, 1) == 0
	// Drain any pending value from a previous firing, so the new
	// firing isn't shadowed by the old one.
	select {
	case <-t.c:
	default:
	}
	atomic.StoreUint32(&t.stopped, 0)
	var dur uint64
	if d > 0 {
		dur = uint64(d)
	}
	go runTimer(t, dur)
	return wasActive
}
