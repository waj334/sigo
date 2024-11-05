package runtime

import (
	"math/rand"
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
	cond   *sync.Cond
	closed bool
	full   bool
}

func channelMake(T *_type, capacity int) _channel {
	channelType := (*_channelTypeData)(T.data)
	var buffer unsafe.Pointer
	if capacity == 0 {
		// Allocate memory for at most one element
		buffer = alloc(uintptr(channelType.elementType.size))
	} else {
		buffer = alloc(uintptr(channelType.elementType.size) * uintptr(capacity))
	}

	return _channel{
		capacity: capacity,
		chanType: T,
		state: &_chanState{
			buffer: buffer,
			cond:   sync.NewCond(new(sync.Mutex)),
		},
	}
}

func channelSend(c _channel, val unsafe.Pointer) {
	channelType := (*_channelTypeData)(c.chanType.data)
	c.state.cond.L.Lock()

	if c.state.closed {
		// Channel is closed, cannot send
		c.state.cond.L.Unlock()
		panic(plainError("send on closed channel"))
	}

	// The channel needs to be able to store at least one item
	actualCap := c.capacity
	if actualCap == 0 {
		actualCap = 1
	}

	nextWriteIndex := (c.state.windex + 1) % actualCap

	// Wait until there is space available in the channel
	for c.state.full {
		if c.state.closed {
			// Channel is closed while waiting, cannot send
			c.state.cond.L.Unlock()
			panic(plainError("send on closed channel"))
		}
		c.state.cond.Wait()
		nextWriteIndex = (c.state.windex + 1) % actualCap
	}

	// Send the value
	ptr := unsafe.Add(c.state.buffer, uintptr(c.state.windex)*uintptr(channelType.elementType.size))
	memcpy(ptr, val, uintptr(channelType.elementType.size))

	// Update write index
	c.state.windex = nextWriteIndex

	// Buffer is now full if writeIndex == readIndex
	c.state.full = c.state.windex == c.state.rindex

	// Signal any goroutines waiting to receive
	c.state.cond.Signal()

	c.state.cond.L.Unlock()
}

func channelReceive(c _channel, block bool) (unsafe.Pointer, bool) {
	if c.state == nil {
		// Block indefinitely.
		for {
			schedulerPause()
		}
	}

	c.state.cond.L.Lock()
	result, ok := _channelReceive(c, block)
	c.state.cond.L.Unlock()
	return result, ok
}

func _channelReceive(c _channel, block bool) (unsafe.Pointer, bool) {
	if c.state == nil {
		// Block indefinitely.
		for {
			schedulerPause()
		}
	}

	channelType := (*_channelTypeData)(c.chanType.data)
	if c.state.rindex == c.state.windex {
		if c.state.closed || !block {
			// Receive the zero value immediately
			return nil, false
		}
	}

	// Block the current _goroutine until there is a value to receive
	for ; c.state.rindex == c.state.windex && !c.state.full && !c.state.closed; c.state.cond.Wait() {
	}

	// Return the zero value if the channel was closed while the current goroutine was waiting.
	if c.state.closed {
		return nil, false
	}

	// Receive the value
	result := unsafe.Add(c.state.buffer, uintptr(c.state.rindex)*uintptr(channelType.elementType.size))

	// The channel needs to be able to store at least one item
	actualCap := c.capacity
	if actualCap == 0 {
		actualCap = 1
	}

	// Advance the read index, wrapping around if necessary
	c.state.rindex = (c.state.rindex + 1) % actualCap
	c.state.full = false
	return result, true
}

func channelClose(c _channel) {
	c.state.closed = true
	c.state.cond.Broadcast()
}

func channelLen(c _channel) int {
	if c.state.rindex < c.state.windex {
		return c.state.windex - c.state.rindex
	}
	return c.state.rindex - c.state.windex
}

func channelCap(c _channel) int {
	return c.capacity
}

func channelRange(c _channel) (unsafe.Pointer, bool) {
	if c.state == nil {
		// Block indefinitely.
		for {
			schedulerPause()
		}
	}

	// Receive the next available value on channel.
	c.state.cond.L.Lock()
	result, ok := _channelReceive(c, true)
	c.state.cond.L.Unlock()
	return result, ok
}

func channelSelect(chanArr *_channel, sendArr *bool, readyArr *int, count int, hasDefault bool) int {
	cc := unsafe.Slice(chanArr, count)
	ss := unsafe.Slice(sendArr, count)
	rdy := unsafe.Slice(readyArr, count)

	for {
		r := 0
		for i := range cc {
			c := cc[i]
			if c.state == nil {
				continue
			}

			send := ss[i]
			c.state.cond.L.Lock()
			if send {
				if !c.state.full {
					// A value can be sent over the channel.
					rdy[r] = i
					r++
				}
			} else if c.state.full || c.state.rindex != c.state.windex {
				// A value can be received from the channel.
				rdy[r] = i
				r++
			}
			c.state.cond.L.Unlock()
		}

		if r == 1 {
			return rdy[0]
		} else if r > 1 {
			return rdy[rand.Intn(r)]
		} else if hasDefault {
			break
		}

		// If no cases are ready and there's no default case, yield to another goroutine.
		schedulerPause()
	}

	return -1
}
