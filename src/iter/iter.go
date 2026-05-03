// Package iter provides Seq, Seq2 type aliases and Pull / Pull2 builders
// for range-over-func iteration.
//
// This is a sigo-specific override of the stdlib iter package. The stdlib
// version uses `type coro struct{}` and binds runtime.newcoro/coroswitch
// via go:linkname. Sigo's MLIR-based compiler is strict about named-type
// identity at call sites, so we use a type alias to runtime.Coro instead.
// The runtime.newcoro / runtime.coroswitch calls use the same Go-level
// types as the runtime declares.
//
// race-detector calls and goexit panic-value propagation are omitted —
// sigo doesn't ship the race detector and runtime.Goexit propagation
// inside iterators is a corner case the embedded use cases don't need.
package iter

import "runtime"

// coro is a type alias to runtime.Coro so that iter's *coro and runtime's
// *coro are the same Go type. The runtime API (newcoro/coroswitch) is
// referenced via the linkname pragmas below.
type coro = runtime.Coro

//go:extern newcoro runtime.newcoro
func newcoro(f func(*coro)) *coro

//go:extern coroswitch runtime.coroswitch
func coroswitch(c *coro)

// Seq is an iterator over sequences of individual values.
type Seq[V any] func(yield func(V) bool)

// Seq2 is an iterator over sequences of pairs of values.
type Seq2[K, V any] func(yield func(K, V) bool)

// Pull converts the "push-style" iterator sequence seq into a "pull-style"
// iterator accessed by the two functions next and stop.
//
// Next returns the next value in the sequence and a boolean indicating
// whether the value is valid. When the sequence is over, next returns the
// zero V and false. It is valid to call next after reaching the end of the
// sequence; subsequent calls return zero V, false.
//
// Stop ends the iteration. It must be called when the caller is no longer
// interested in next values. It is valid to call stop multiple times and
// when next has not yet signaled the end of the sequence.
func Pull[V any](seq Seq[V]) (next func() (V, bool), stop func()) {
	var pull struct {
		v    V
		ok   bool
		done bool // either the sequence is exhausted or stop was called
	}
	var c *coro

	c = newcoro(func(self *coro) {
		if pull.done {
			return
		}
		yield := func(v1 V) bool {
			if pull.done {
				return false
			}
			pull.v, pull.ok = v1, true
			coroswitch(self)
			return !pull.done
		}
		seq(yield)
		var zero V
		pull.v, pull.ok = zero, false
		pull.done = true
	})

	next = func() (v1 V, ok1 bool) {
		if pull.done {
			return
		}
		coroswitch(c)
		return pull.v, pull.ok
	}

	stop = func() {
		if pull.done {
			return
		}
		pull.done = true
		coroswitch(c)
	}

	return next, stop
}

// Pull2 converts the "push-style" iterator sequence seq into a "pull-style"
// iterator accessed by the two functions next and stop.
func Pull2[K, V any](seq Seq2[K, V]) (next func() (K, V, bool), stop func()) {
	var pull struct {
		k    K
		v    V
		ok   bool
		done bool
	}
	var c *coro

	c = newcoro(func(self *coro) {
		if pull.done {
			return
		}
		yield := func(k1 K, v1 V) bool {
			if pull.done {
				return false
			}
			pull.k, pull.v, pull.ok = k1, v1, true
			coroswitch(self)
			return !pull.done
		}
		seq(yield)
		var zeroK K
		var zeroV V
		pull.k, pull.v, pull.ok = zeroK, zeroV, false
		pull.done = true
	})

	next = func() (k1 K, v1 V, ok1 bool) {
		if pull.done {
			return
		}
		coroswitch(c)
		return pull.k, pull.v, pull.ok
	}

	stop = func() {
		if pull.done {
			return
		}
		pull.done = true
		coroswitch(c)
	}

	return next, stop
}
