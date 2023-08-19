package runtime

import (
	"math/rand"
	"sync"
	"unsafe"
)

type _channel struct {
	buffer     unsafe.Pointer
	capacity   int
	readIndex  *int
	writeIndex *int
	chanType   *_type
	direction  int
	cond       *sync.Cond
	closed     *bool
	full       *bool
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

	read := 0
	write := 0
	closed := false
	full := false
	mu := new(sync.Mutex)

	return _channel{
		buffer:     buffer,
		capacity:   capacity,
		readIndex:  &read,
		writeIndex: &write,
		chanType:   T,
		cond:       sync.NewCond(mu),
		closed:     &closed,
		full:       &full,
	}
}

func channelSend(c _channel, val unsafe.Pointer) {
	channelType := (*_channelTypeData)(c.chanType.data)
	c.cond.L.Lock()

	if *c.closed {
		// Channel is closed, cannot send
		c.cond.L.Unlock()
		panic(plainError("send on closed channel"))
	}

	// The channel needs to be able to store at least one item
	actualCap := c.capacity
	if actualCap == 0 {
		actualCap = 1
	}

	nextWriteIndex := (*c.writeIndex + 1) % actualCap

	// Wait until there is space available in the channel
	for *c.full {
		if *c.closed {
			// Channel is closed while waiting, cannot send
			c.cond.L.Unlock()
			panic(plainError("send on closed channel"))
		}
		c.cond.Wait()
		nextWriteIndex = (*c.writeIndex + 1) % actualCap
	}

	// Send the value
	ptr := unsafe.Add(c.buffer, uintptr(*c.writeIndex)*uintptr(channelType.elementType.size))
	memcpy(ptr, val, uintptr(channelType.elementType.size))

	// Update write index
	*c.writeIndex = nextWriteIndex

	// Buffer is now full if writeIndex == readIndex
	*c.full = *c.writeIndex == *c.readIndex

	// Signal any goroutines waiting to receive
	c.cond.Signal()

	c.cond.L.Unlock()
}

func channelReceive(c _channel, result unsafe.Pointer, block bool) (ok bool) {
	channelType := (*_channelTypeData)(c.chanType.data)

	// Zero the result storage location.
	memset(result, 0, uintptr(channelType.elementType.size))

	// Attempt to acquire the channel's lock.
	c.cond.L.Lock()

	if c.readIndex == c.writeIndex {
		if *c.closed || !block {
			// Receive the zero value immediately
			c.cond.L.Unlock()
			return false
		}
	}

	// Block the current _goroutine until there is a value to receive
	for ; *c.readIndex == *c.writeIndex && !*c.full && !*c.closed; c.cond.Wait() {
	}

	// Return the zero value if the channel was closed while the current goroutine was waiting.
	if *c.closed {
		c.cond.L.Unlock()
		return false
	}

	// Receive the value
	if result != nil {
		ptr := unsafe.Add(c.buffer, uintptr(*c.readIndex)*uintptr(channelType.elementType.size))
		memcpy(result, ptr, uintptr(channelType.elementType.size))
	}

	// The channel needs to be able to store at least one item
	actualCap := c.capacity
	if actualCap == 0 {
		actualCap = 1
	}

	// Advance the read index, wrapping around if necessary
	*c.readIndex = (*c.readIndex + 1) % actualCap

	*c.full = false

	c.cond.L.Unlock()
	return true
}

func channelClose(c _channel) {
	*c.closed = true
	c.cond.Broadcast()
}

func channelLen(c _channel) int {
	if *c.readIndex < *c.writeIndex {
		return *c.writeIndex - *c.readIndex
	}
	return *c.readIndex - *c.writeIndex
}

func channelCap(c _channel) int {
	return c.capacity
}

func channelSelect(_chans, _send, _readyCases _slice, hasDefault bool) int {
	chans := *(*[]_channel)(unsafe.Pointer(&_chans))
	send := *(*[]bool)(unsafe.Pointer(&_send))
	readyCases := *(*[]int)(unsafe.Pointer(&_readyCases))
	ii := 0

	for {
		// Reset readyCases counter.
		ii = 0

		// Check which cases are ready.
		for i, c := range chans {
			// TODO: The default case will introduce an invalid channel. Improve the builder to remove this from the
			//       input slices.
			if c.buffer != nil {
				c.cond.L.Lock()
				switch {
				case send[i]:
					if !*c.full {
						readyCases[ii] = i
						ii++
					}
				default:
					if *c.readIndex != *c.writeIndex || *c.full {
						readyCases[ii] = i
						ii++
					}
				}
				c.cond.L.Unlock()
			}
		}

		// If one or more cases are ready, choose one at random.
		if ii > 0 {
			randIndex := rand.Intn(ii)
			caseIndex := readyCases[randIndex]
			return caseIndex
		} else if hasDefault {
			// If no cases are ready and there's a default case, execute it.
			return len(chans)
		}

		// If no cases are ready and there's no default case, yield to another goroutine.
		schedulerPause()
	}
}

func channelRange(_chan _channel) unsafe.Pointer {
	panic("todo")
}

func _channelSelect(_cases, _results, _send, _values, _readyCases _slice, hasDefault bool) (int, bool) {
	cases := *(*[]_channel)(unsafe.Pointer(&_cases))
	results := *(*[]unsafe.Pointer)(unsafe.Pointer(&_results))
	send := *(*[]bool)(unsafe.Pointer(&_send))
	values := *(*[]unsafe.Pointer)(unsafe.Pointer(&_values))
	readyCases := *(*[]int)(unsafe.Pointer(&_readyCases))
	ii := 0

	for {
		// Reset readyCases counter.
		ii = 0

		// Check which cases are ready.
		for i, c := range cases {
			c.cond.L.Lock()
			switch {
			case send[i]:
				if !*c.full {
					readyCases[ii] = i
					ii++
				}
			default:
				if *c.readIndex != *c.writeIndex || *c.full {
					readyCases[ii] = i
					ii++
				}
			}
			c.cond.L.Unlock()
		}

		// If one or more cases are ready, choose one at random.
		if ii > 0 {
			randIndex := rand.Intn(ii)
			caseIndex := readyCases[randIndex]
			ok := true

			// Perform the send or receive operation.
			if send[caseIndex] {
				channelSend(cases[caseIndex], values[caseIndex])
			} else {
				ok = channelReceive(cases[caseIndex], results[caseIndex], true)
			}
			return caseIndex, ok
		} else if hasDefault {
			// If no cases are ready and there's a default case, execute it.
			return len(cases), false
		}

		// If no cases are ready and there's no default case, yield to another goroutine.
		schedulerPause()
	}
}
