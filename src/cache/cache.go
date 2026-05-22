package cache

import "unsafe"

type CacheOps interface {
	Clean(addr unsafe.Pointer, size uintptr)
	Invalidate(addr unsafe.Pointer, size uintptr)
	CleanInvalidate(addr unsafe.Pointer, size uintptr)
}

// NoCache is a zero-sized CacheOps implementation that performs no
// maintenance. Use it for buffers in non-cacheable memory regions.
type NoCache struct{}

func (NoCache) Clean(addr unsafe.Pointer, size uintptr)           {}
func (NoCache) Invalidate(addr unsafe.Pointer, size uintptr)      {}
func (NoCache) CleanInvalidate(addr unsafe.Pointer, size uintptr) {}
