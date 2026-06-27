package mailbox

import "runtime"

type Storage[T any] interface {
	Len() int
	Ptr(index int) *T
}

type Slice[T any] []T

func (s Slice[T]) Len() int {
	return len(s)
}

func (s Slice[T]) Ptr(index int) *T {
	return &s[index]
}

type OverflowPolicy uint8

const (
	DropNewest OverflowPolicy = iota
	DropOldest
	Coalesce
)

type Mailbox[T any, StorageT Storage[T]] struct {
	storage StorageT

	rindex int
	windex int
	count  int

	policy OverflowPolicy

	dropped uint32
	inited  bool
}

func (m *Mailbox[T, StorageT]) Init(storage StorageT, policy OverflowPolicy) {
	if storage.Len() <= 0 {
		panic("mailbox: empty storage")
	}

	state := runtime.DisableInterrupts()

	m.storage = storage
	m.rindex = 0
	m.windex = 0
	m.count = 0
	m.policy = policy
	m.dropped = 0
	m.inited = true

	runtime.EnableInterrupts(state)
}

func (m *Mailbox[T, StorageT]) Cap() int {
	if !m.inited {
		return 0
	}

	return m.storage.Len()
}

func (m *Mailbox[T, StorageT]) Len() int {
	if !m.inited {
		return 0
	}

	state := runtime.DisableInterrupts()
	n := m.count
	runtime.EnableInterrupts(state)

	return n
}

func (m *Mailbox[T, StorageT]) Dropped() uint32 {
	if !m.inited {
		return 0
	}

	state := runtime.DisableInterrupts()
	n := m.dropped
	runtime.EnableInterrupts(state)

	return n
}

// TrySend attempts to enqueue v without blocking.
//
// Safe from interrupt context.
// Does not allocate.
// Does not call the scheduler.
// Does not take a mutex.
func (m *Mailbox[T, StorageT]) TrySend(v T) bool {
	if !m.inited {
		panic("mailbox: use before Init")
	}

	state := runtime.DisableInterrupts()

	ok := m.trySendLocked(v)

	runtime.EnableInterrupts(state)
	return ok
}

// Send enqueues v, yielding until space is available.
//
// Not safe from interrupt context.
func (m *Mailbox[T, StorageT]) Send(v T) {
	if runtime.InInterrupt() {
		panic("mailbox: blocking send from interrupt context")
	}

	for !m.TrySend(v) {
		runtime.Gosched()
	}
}

// TryRecv attempts to dequeue one value without blocking.
//
// Safe from interrupt context.
func (m *Mailbox[T, StorageT]) TryRecv() (T, bool) {
	var zero T

	if !m.inited {
		panic("mailbox: use before Init")
	}

	state := runtime.DisableInterrupts()

	if m.count == 0 {
		runtime.EnableInterrupts(state)
		return zero, false
	}

	v := *m.storage.Ptr(m.rindex)

	capacity := m.storage.Len()
	m.rindex = (m.rindex + 1) % capacity
	m.count--

	runtime.EnableInterrupts(state)
	return v, true
}

// Recv dequeues one value, yielding until one is available.
//
// Not safe from interrupt context.
func (m *Mailbox[T, StorageT]) Recv() T {
	if runtime.InInterrupt() {
		panic("mailbox: blocking receive from interrupt context")
	}

	for {
		if v, ok := m.TryRecv(); ok {
			return v
		}

		runtime.Gosched()
	}
}

func (m *Mailbox[T, StorageT]) Clear() {
	if !m.inited {
		return
	}

	state := runtime.DisableInterrupts()

	m.rindex = 0
	m.windex = 0
	m.count = 0

	runtime.EnableInterrupts(state)
}

func (m *Mailbox[T, StorageT]) trySendLocked(v T) bool {
	capacity := m.storage.Len()

	if m.count == capacity {
		switch m.policy {
		case DropNewest:
			m.dropped++
			return false

		case DropOldest:
			m.dropped++
			m.rindex = (m.rindex + 1) % capacity
			m.count--

		case Coalesce:
			m.dropped++
			return true

		default:
			m.dropped++
			return false
		}
	}

	*m.storage.Ptr(m.windex) = v
	m.windex = (m.windex + 1) % capacity
	m.count++

	return true
}
