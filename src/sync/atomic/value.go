// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import "unsafe"

// A Value provides an atomic load and store of a consistently typed value.
// The zero value for a Value returns nil from [Value.Load].
// Once [Value.Store] has been called, a Value must not be copied.
//
// A Value must not be copied after first use.
type Value struct {
	_ noCopy
	v any
}

// ifaceWords mirrors the in-memory layout of an empty interface as defined by
// runtime._interface: data pointer first, type pointer second. Keep field
// order in sync with src/runtime/interface.go.
type ifaceWords struct {
	data unsafe.Pointer
	typ  unsafe.Pointer
}

// Load returns the value set by the most recent Store.
// It returns nil if there has been no call to Store for this Value.
func (v *Value) Load() (val any) {
	state := disableInterrupts()
	val = v.v
	enableInterrupts(state)
	return
}

// Store sets the value of the [Value] v to val.
// All calls to Store for a given Value must use values of the same concrete type.
// Store of an inconsistent type panics, as does Store(nil).
func (v *Value) Store(val any) {
	if val == nil {
		panic("sync/atomic: store of nil value into Value")
	}
	np := (*ifaceWords)(unsafe.Pointer(&val))
	state := disableInterrupts()
	vp := (*ifaceWords)(unsafe.Pointer(&v.v))
	if vp.typ != nil && vp.typ != np.typ {
		enableInterrupts(state)
		panic("sync/atomic: store of inconsistently typed value into Value")
	}
	v.v = val
	enableInterrupts(state)
}

// Swap stores new into [Value] and returns the previous value. It returns nil
// if the [Value] is empty.
//
// All calls to Swap for a given Value must use values of the same concrete
// type. Swap of an inconsistent type panics, as does Swap(nil).
func (v *Value) Swap(new any) (old any) {
	if new == nil {
		panic("sync/atomic: swap of nil value into Value")
	}
	np := (*ifaceWords)(unsafe.Pointer(&new))
	state := disableInterrupts()
	vp := (*ifaceWords)(unsafe.Pointer(&v.v))
	if vp.typ != nil && vp.typ != np.typ {
		enableInterrupts(state)
		panic("sync/atomic: swap of inconsistently typed value into Value")
	}
	old = v.v
	v.v = new
	enableInterrupts(state)
	return
}

// CompareAndSwap executes the compare-and-swap operation for the [Value].
//
// All calls to CompareAndSwap for a given Value must use values of the same
// concrete type. CompareAndSwap of an inconsistent type panics, as does
// CompareAndSwap(old, nil).
func (v *Value) CompareAndSwap(old, new any) (swapped bool) {
	if new == nil {
		panic("sync/atomic: compare and swap of nil value into Value")
	}
	np := (*ifaceWords)(unsafe.Pointer(&new))
	op := (*ifaceWords)(unsafe.Pointer(&old))
	if op.typ != nil && np.typ != op.typ {
		panic("sync/atomic: compare and swap of inconsistently typed values")
	}
	state := disableInterrupts()
	vp := (*ifaceWords)(unsafe.Pointer(&v.v))
	if vp.typ != nil && vp.typ != np.typ {
		enableInterrupts(state)
		panic("sync/atomic: compare and swap of inconsistently typed value into Value")
	}
	// Runtime equality check between current stored value and old. Matches
	// upstream semantics — for value types this is a deep equality compare,
	// for pointer types it is reference equality.
	if v.v != old {
		enableInterrupts(state)
		return false
	}
	v.v = new
	enableInterrupts(state)
	return true
}
