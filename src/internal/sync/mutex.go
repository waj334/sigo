// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sync provides basic synchronization primitives such as mutual
// exclusion locks to internal packages (including ones that depend on sync).
//
// This is a simplified SiGo implementation for cooperative schedulers.
package sync

import "sync/atomic"

// A Mutex is a mutual exclusion lock.
// The zero value for a Mutex is an unlocked mutex.
type Mutex struct {
	state int32
}

// Lock locks m.
func (m *Mutex) Lock() {
	for !atomic.CompareAndSwapInt32(&m.state, 0, 1) {
	}
}

// TryLock tries to lock m and reports whether it succeeded.
func (m *Mutex) TryLock() bool {
	return atomic.CompareAndSwapInt32(&m.state, 0, 1)
}

// Unlock unlocks m.
func (m *Mutex) Unlock() {
	atomic.StoreInt32(&m.state, 0)
}
