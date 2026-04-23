package atomic

import "unsafe"

// A Bool is an atomic boolean value.
// The zero value is false.
type Bool struct {
	_ noCopy
	v uint32
}

// Load atomically loads and returns the value stored in x.
func (x *Bool) Load() bool {
	return LoadUint32(&x.v) != 0
}

// Store atomically stores val into x.
func (x *Bool) Store(val bool) {
	StoreUint32(&x.v, b32(val))
}

// Swap atomically stores new into x and returns the previous value.
func (x *Bool) Swap(new bool) (old bool) {
	return SwapUint32(&x.v, b32(new)) != 0
}

// CompareAndSwap executes the compare-and-swap operation for the boolean value x.
func (x *Bool) CompareAndSwap(old, new bool) (swapped bool) {
	return CompareAndSwapUint32(&x.v, b32(old), b32(new))
}

// b32 returns a uint32 0 or 1 representing b.
func b32(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// A Pointer is an atomic pointer of type *T. The zero value is a nil *T.
type Pointer[T any] struct {
	_ [0]*T
	_ noCopy
	v unsafe.Pointer
}

// Load atomically loads and returns the value stored in x.
func (x *Pointer[T]) Load() *T {
	return (*T)(LoadPointer(&x.v))
}

// Store atomically stores val into x.
func (x *Pointer[T]) Store(val *T) {
	StorePointer(&x.v, unsafe.Pointer(val))
}

// Swap atomically stores new into x and returns the previous value.
func (x *Pointer[T]) Swap(new *T) (old *T) {
	return (*T)(SwapPointer(&x.v, unsafe.Pointer(new)))
}

// CompareAndSwap executes the compare-and-swap operation for x.
func (x *Pointer[T]) CompareAndSwap(old, new *T) (swapped bool) {
	return CompareAndSwapPointer(&x.v, unsafe.Pointer(old), unsafe.Pointer(new))
}

// An Int32 is an atomic int32. The zero value is zero.
type Int32 struct {
	_ noCopy
	v int32
}

func (x *Int32) Load() int32                        { return LoadInt32(&x.v) }
func (x *Int32) Store(val int32)                    { StoreInt32(&x.v, val) }
func (x *Int32) Swap(new int32) (old int32)         { return SwapInt32(&x.v, new) }
func (x *Int32) Add(delta int32) (new int32)        { return AddInt32(&x.v, delta) }
func (x *Int32) CompareAndSwap(old, new int32) bool { return CompareAndSwapInt32(&x.v, old, new) }

// An Int64 is an atomic int64. The zero value is zero.
type Int64 struct {
	_ noCopy
	_ align64
	v int64
}

func (x *Int64) Load() int64                        { return LoadInt64(&x.v) }
func (x *Int64) Store(val int64)                    { StoreInt64(&x.v, val) }
func (x *Int64) Swap(new int64) (old int64)         { return SwapInt64(&x.v, new) }
func (x *Int64) Add(delta int64) (new int64)        { return AddInt64(&x.v, delta) }
func (x *Int64) CompareAndSwap(old, new int64) bool { return CompareAndSwapInt64(&x.v, old, new) }

// A Uint32 is an atomic uint32. The zero value is zero.
type Uint32 struct {
	_ noCopy
	v uint32
}

func (x *Uint32) Load() uint32                        { return LoadUint32(&x.v) }
func (x *Uint32) Store(val uint32)                    { StoreUint32(&x.v, val) }
func (x *Uint32) Swap(new uint32) (old uint32)        { return SwapUint32(&x.v, new) }
func (x *Uint32) Add(delta uint32) (new uint32)       { return AddUint32(&x.v, delta) }
func (x *Uint32) CompareAndSwap(old, new uint32) bool { return CompareAndSwapUint32(&x.v, old, new) }

// A Uint64 is an atomic uint64. The zero value is zero.
type Uint64 struct {
	_ noCopy
	_ align64
	v uint64
}

func (x *Uint64) Load() uint64                        { return LoadUint64(&x.v) }
func (x *Uint64) Store(val uint64)                    { StoreUint64(&x.v, val) }
func (x *Uint64) Swap(new uint64) (old uint64)        { return SwapUint64(&x.v, new) }
func (x *Uint64) Add(delta uint64) (new uint64)       { return AddUint64(&x.v, delta) }
func (x *Uint64) CompareAndSwap(old, new uint64) bool { return CompareAndSwapUint64(&x.v, old, new) }

// A Uintptr is an atomic uintptr. The zero value is zero.
type Uintptr struct {
	_ noCopy
	v uintptr
}

func (x *Uintptr) Load() uintptr                        { return LoadUintptr(&x.v) }
func (x *Uintptr) Store(val uintptr)                    { StoreUintptr(&x.v, val) }
func (x *Uintptr) Swap(new uintptr) (old uintptr)       { return SwapUintptr(&x.v, new) }
func (x *Uintptr) Add(delta uintptr) (new uintptr)      { return AddUintptr(&x.v, delta) }
func (x *Uintptr) CompareAndSwap(old, new uintptr) bool { return CompareAndSwapUintptr(&x.v, old, new) }

// noCopy may be added to structs which must not be copied
// after the first use.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// align64 may be added to structs that must be 64-bit aligned.
// This struct is recognized by a special case in the compiler
// and will not work if copied to a different package.
type align64 struct{}
