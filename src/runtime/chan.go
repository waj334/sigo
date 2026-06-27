package runtime

import (
	"sync"
	"unsafe"
)

type _channel struct {
	capacity int
	chanType *_type
	state    *_chanState
}

type _chanState struct {
	buffer unsafe.Pointer

	rindex int
	windex int
	count  int

	cond *sync.Cond
	mu   sync.Mutex

	closed bool

	// Used only for capacity == 0 rendezvous channels.
	//
	// full means a sender has placed a value in buffer and is waiting for a
	// receiver to consume it. sendSeq/deliveredSeq let a sender distinguish:
	//
	//   - my value was received, even if the channel was closed afterward
	//   - channel was closed before my value was received
	//
	full         bool
	sendSeq      uint32
	deliveredSeq uint32
	recvWaiting  int
}

func channelMake(T *_type, capacity int) _channel {
	if capacity < 0 {
		panic(plainError("makechan: negative channel capacity"))
	}

	channelType := (*_channelTypeData)(T.data)
	elemSize := uintptr(channelType.elementType.size)
	if elemSize == 0 {
		elemSize = 1
	}

	var buffer unsafe.Pointer
	if capacity == 0 {
		// One rendezvous slot. This is not channel capacity; it is only the
		// temporary handoff storage between a blocked sender and receiver.
		buffer = alloc(elemSize)
	} else {
		buffer = alloc(elemSize * uintptr(capacity))
	}

	state := &_chanState{
		buffer: buffer,
	}

	state.cond = sync.NewCond(&state.mu)

	return _channel{
		capacity: capacity,
		chanType: T,
		state:    state,
	}
}

func channelMustNotBeInInterrupt() {
	if InInterrupt() {
		panic(plainError("channel operation from interrupt context"))
	}
}

func channelElemSize(c _channel) uintptr {
	channelType := (*_channelTypeData)(c.chanType.data)
	elemSize := uintptr(channelType.elementType.size)
	if elemSize == 0 {
		return 1
	}
	return elemSize
}

func channelSend(c _channel, val unsafe.Pointer) {
	channelMustNotBeInInterrupt()

	if c.state == nil {
		// Send on nil channel blocks forever.
		for {
			gosched()
		}
	}

	s := c.state
	s.cond.L.Lock()

	if c.capacity == 0 {
		channelSendUnbuffered(c, val)
		s.cond.L.Unlock()
		return
	}

	channelSendBuffered(c, val)

	s.cond.L.Unlock()
}

func channelSendBuffered(c _channel, val unsafe.Pointer) {
	s := c.state
	elemSize := channelElemSize(c)

	if s.closed {
		panic(plainError("send on closed channel"))
	}

	for s.count == c.capacity {
		if s.closed {
			panic(plainError("send on closed channel"))
		}

		s.cond.Wait()
	}

	if s.closed {
		panic(plainError("send on closed channel"))
	}

	dst := unsafe.Add(s.buffer, uintptr(s.windex)*elemSize)
	memcpy(dst, val, elemSize)

	s.windex = (s.windex + 1) % c.capacity
	s.count++

	// Wake receivers.
	s.cond.Broadcast()
}

func channelSendUnbuffered(c _channel, val unsafe.Pointer) {
	s := c.state
	elemSize := channelElemSize(c)

	if s.closed {
		panic(plainError("send on closed channel"))
	}

	// Only one sender may occupy the rendezvous slot at a time.
	for s.full {
		if s.closed {
			panic(plainError("send on closed channel"))
		}

		s.cond.Wait()
	}

	if s.closed {
		panic(plainError("send on closed channel"))
	}

	s.sendSeq++
	mySeq := s.sendSeq

	memcpy(s.buffer, val, elemSize)
	s.full = true

	// Wake receivers.
	s.cond.Broadcast()

	// A zero-capacity send does not complete until a receiver consumes the
	// value.
	for s.full && !s.closed {
		s.cond.Wait()
	}

	// If a receiver consumed this sender's value, the send succeeds even if
	// another goroutine closed the channel before this sender reacquired the
	// lock.
	if s.deliveredSeq == mySeq {
		return
	}

	// Otherwise the channel was closed before rendezvous completed.
	panic(plainError("send on closed channel"))
}

func channelReceive(c _channel, block bool) (unsafe.Pointer, bool) {
	channelMustNotBeInInterrupt()

	if c.state == nil {
		if !block {
			return nil, false
		}

		// Receive from nil channel blocks forever.
		for {
			gosched()
		}
	}

	s := c.state
	s.cond.L.Lock()

	var result unsafe.Pointer
	var ok bool

	if c.capacity == 0 {
		result, ok = channelReceiveUnbuffered(c, block)
	} else {
		result, ok = channelReceiveBuffered(c, block)
	}

	s.cond.L.Unlock()
	return result, ok
}

func channelReceiveBuffered(c _channel, block bool) (unsafe.Pointer, bool) {
	s := c.state
	elemSize := channelElemSize(c)

	for s.count == 0 {
		if s.closed {
			return nil, false
		}

		if !block {
			return nil, false
		}

		s.cond.Wait()
	}

	src := unsafe.Add(s.buffer, uintptr(s.rindex)*elemSize)

	s.rindex = (s.rindex + 1) % c.capacity
	s.count--

	// Wake blocked senders.
	s.cond.Broadcast()

	return src, true
}

func channelReceiveUnbuffered(c _channel, block bool) (unsafe.Pointer, bool) {
	s := c.state

	if !s.full {
		if s.closed {
			return nil, false
		}

		if !block {
			return nil, false
		}

		s.recvWaiting++

		for !s.full && !s.closed {
			s.cond.Wait()
		}

		s.recvWaiting--
	}

	if !s.full {
		// Closed while waiting.
		return nil, false
	}

	result := s.buffer

	s.deliveredSeq = s.sendSeq
	s.full = false

	// Wake the sender that completed rendezvous, plus any other senders that
	// were waiting for the rendezvous slot.
	s.cond.Broadcast()

	return result, true
}

func channelClose(c _channel) {
	channelMustNotBeInInterrupt()

	if c.state == nil {
		panic(plainError("close of nil channel"))
	}

	s := c.state
	s.cond.L.Lock()

	if s.closed {
		s.cond.L.Unlock()
		panic(plainError("close of closed channel"))
	}

	s.closed = true

	if c.capacity == 0 {
		// If a sender was waiting for rendezvous completion, cancel that
		// pending handoff. The sender will wake and panic.
		s.full = false
	}

	s.cond.Broadcast()
	s.cond.L.Unlock()
}

func channelLen(c _channel) int {
	if c.state == nil {
		return 0
	}

	s := c.state
	s.cond.L.Lock()

	var n int
	if c.capacity == 0 {
		// Go's len on an unbuffered channel is always zero.
		n = 0
	} else {
		n = s.count
	}

	s.cond.L.Unlock()
	return n
}

func channelCap(c _channel) int {
	return c.capacity
}

func channelRange(c _channel) (unsafe.Pointer, bool) {
	return channelReceive(c, true)
}

func channelSelect(chanArr *_channel, sendArr *bool, readyArr *int, count int, hasDefault bool) int {
	channelMustNotBeInInterrupt()

	cc := unsafe.Slice(chanArr, count)
	ss := unsafe.Slice(sendArr, count)
	rdy := unsafe.Slice(readyArr, count)

	for {
		readyCount := 0

		for i := range cc {
			c := cc[i]
			if c.state == nil {
				continue
			}

			s := c.state
			s.cond.L.Lock()

			send := ss[i]

			if send {
				if s.closed {
					// A send on a closed channel is immediately selected and
					// then panics when the send operation runs.
					rdy[readyCount] = i
					readyCount++
				} else if c.capacity == 0 {
					// Approximation: unbuffered send is ready if a receiver is
					// already blocked.
					if s.recvWaiting > 0 && !s.full {
						rdy[readyCount] = i
						readyCount++
					}
				} else if s.count < c.capacity {
					rdy[readyCount] = i
					readyCount++
				}
			} else {
				if c.capacity == 0 {
					if s.full || s.closed {
						rdy[readyCount] = i
						readyCount++
					}
				} else {
					if s.count > 0 || s.closed {
						rdy[readyCount] = i
						readyCount++
					}
				}
			}

			s.cond.L.Unlock()
		}

		if readyCount == 1 {
			return rdy[0]
		}

		if readyCount > 1 {
			return rdy[randn(uint32(readyCount))]
		}

		if hasDefault {
			return -1
		}

		gosched()
	}
}

func channelIsNil(c _channel) bool {
	return c.state == nil
}
