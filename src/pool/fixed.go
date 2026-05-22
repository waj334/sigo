// Package pool provides fixed-region allocators suitable for DMA buffers.
//
// The backing slice supplied to the constructor must reside in DMA-reachable
// memory. Cache coherency between CPU and DMA engine is the caller's
// responsibility, expressed through the cache.CacheOps type parameter:
// pass runtime.NoCache for non-cacheable regions, or a platform-specific
// cache implementation for cacheable regions.
package pool

import (
	"cache"
	"sync/atomic"
	"unsafe"
)

// freeNode is the intrusive free-list link stored in unallocated slots.
// The first sizeof(uintptr) bytes of every free slot holds the address of
// the next free slot, or nil at the end of the list.
type freeNode struct {
	next unsafe.Pointer
}

// FixedPool is a lock-free, fixed-slot-size allocator backed by a
// caller-supplied byte slice.
//
// Get and Put are safe for concurrent use across goroutines and ISRs on
// a single core. For AMP configurations sharing a pool across cores,
// external synchronization (e.g., HSEM) is required and the free-list
// head should reside in non-cacheable memory.
type FixedPool[C cache.CacheOps] struct {
	base     unsafe.Pointer
	slotSize uintptr
	capacity uintptr
	free     atomic.Pointer[freeNode]
	inUse    atomic.Int32
	peak     atomic.Int32
	ops      C
}

// Stats holds runtime allocation statistics.
type Stats struct {
	Capacity int
	InUse    int
	Free     int
	Peak     int
}

// NewFixed constructs a FixedPool over buf, dividing it into slots of
// slotSize bytes (rounded up to align). The pool's capacity is determined
// by how many aligned slots fit in buf after the start address is rounded
// up to align.
//
// Align must be a non-zero power of two. For DMA buffers on cores with
// data caches, align should be at least the cache line size, and slotSize
// should be a multiple of that to prevent partial cache line sharing
// between adjacent slots.
//
//go:nowritebarrier
func NewFixed[C cache.CacheOps](
	buf []byte,
	slotSize, align uintptr,
) (*FixedPool[C], error) {
	if align == 0 || align&(align-1) != 0 {
		return nil, ErrInvalidAlignment
	}
	if slotSize == 0 {
		return nil, ErrInvalidSlotSize
	}
	if uintptr(len(buf)) < align {
		return nil, ErrBufferTooSmall
	}

	// Round slot size up to the alignment boundary so consecutive slots
	// remain aligned.
	slotSize = (slotSize + align - 1) &^ (align - 1)

	// A slot must hold the free-list link.
	if slotSize < unsafe.Sizeof(freeNode{}) {
		slotSize = (unsafe.Sizeof(freeNode{}) + align - 1) &^ (align - 1)
	}

	// Align the base pointer up.
	raw := uintptr(unsafe.Pointer(&buf[0]))
	aligned := (raw + align - 1) &^ (align - 1)
	offset := aligned - raw
	if offset >= uintptr(len(buf)) {
		return nil, ErrBufferTooSmall
	}
	usable := uintptr(len(buf)) - offset
	capacity := usable / slotSize
	if capacity == 0 {
		return nil, ErrBufferTooSmall
	}

	p := &FixedPool[C]{
		base:     unsafe.Pointer(aligned),
		slotSize: slotSize,
		capacity: capacity,
	}

	// Build the free list. Walk slots in reverse so the head ends up
	// pointing at slot 0, giving sequential first allocations good
	// spatial locality.
	var head *freeNode
	for i := int(capacity) - 1; i >= 0; i-- {
		slot := (*freeNode)(unsafe.Add(p.base, uintptr(i)*slotSize))
		slot.next = unsafe.Pointer(head)
		head = slot
	}
	p.free.Store(head)

	return p, nil
}

// MustNewFixed is like NewFixed but panics on error. Convenient for
// package-level pool initialization where misconfiguration is a bug.
func MustNewFixed[C cache.CacheOps](
	buf []byte,
	slotSize, align uintptr,
) *FixedPool[C] {
	p, err := NewFixed[C](buf, slotSize, align)
	if err != nil {
		panic(err)
	}
	return p
}

// Get returns a slice referencing one slot or nil if the pool is exhausted.
// The returned slice has len == cap == SlotSize(). Contents are unspecified
// — callers must overwrite before reading.
func (p *FixedPool[C]) Get() []byte {
	for {
		head := p.free.Load()
		if head == nil {
			return nil
		}
		next := (*freeNode)(head.next)
		if p.free.CompareAndSwap(head, next) {
			n := p.inUse.Add(1)
			// Update peak; relaxed CAS is fine for a stat.
			for {
				peak := p.peak.Load()
				if n <= peak || p.peak.CompareAndSwap(peak, n) {
					break
				}
			}
			return unsafe.Slice((*byte)(unsafe.Pointer(head)), p.slotSize)
		}
	}
}

// Put returns a slot to the pool. The slice must have been obtained from
// Get on the same pool; passing any other slice produces undefined behavior.
func (p *FixedPool[C]) Put(b []byte) {
	if len(b) == 0 {
		return
	}
	slot := (*freeNode)(unsafe.Pointer(&b[0]))
	for {
		head := p.free.Load()
		slot.next = unsafe.Pointer(head)
		if p.free.CompareAndSwap(head, slot) {
			if p.inUse.Add(-1) < 0 {
				panic("invalid put")
			}
			return
		}
	}
}

// PrepareTx performs cache maintenance required before a DMA engine reads
// from b. For cacheable buffers, this writes back any dirty cache lines.
// No-op when C is runtime.NoCache.
func (p *FixedPool[C]) PrepareTx(b []byte) {
	if len(b) == 0 {
		return
	}
	p.ops.Clean(unsafe.Pointer(&b[0]), uintptr(len(b)))
}

// CompleteRx performs cache maintenance required after a DMA engine has
// written to b. For cacheable buffers, this invalidates any cache lines
// covering b so later CPU reads observe the DMA-written data. No-op
// when C is runtime.NoCache.
func (p *FixedPool[C]) CompleteRx(b []byte) {
	if len(b) == 0 {
		return
	}
	p.ops.Invalidate(unsafe.Pointer(&b[0]), uintptr(len(b)))
}

// PrepareBidir performs cache maintenance for buffers the DMA engine will
// both read and write. No-op when C is runtime.NoCache.
func (p *FixedPool[C]) PrepareBidir(b []byte) {
	if len(b) == 0 {
		return
	}
	p.ops.CleanInvalidate(unsafe.Pointer(&b[0]), uintptr(len(b)))
}

// SlotSize returns the size of one slot in bytes. May exceed the requested
// slot size due to alignment rounding.
func (p *FixedPool[C]) SlotSize() uintptr { return p.slotSize }

// Cap returns the total number of slots in the pool.
func (p *FixedPool[C]) Cap() int { return int(p.capacity) }

// Stats returns a snapshot of current allocation statistics.
func (p *FixedPool[C]) Stats() Stats {
	inUse := int(p.inUse.Load())
	return Stats{
		Capacity: int(p.capacity),
		InUse:    inUse,
		Free:     int(p.capacity) - inUse,
		Peak:     int(p.peak.Load()),
	}
}
