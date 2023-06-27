package runtime

import (
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

func channelMake(chanTyp *_type, capacity int) _channel {
	var buffer unsafe.Pointer
	if capacity == 0 {
		// Allocate memory for at most one element
		buffer = alloc(chanTyp.channel.elementType.size)
	} else {
		buffer = alloc(chanTyp.channel.elementType.size * uintptr(capacity))
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
		chanType:   chanTyp,
		cond:       sync.NewCond(mu),
		closed:     &closed,
		full:       &full,
	}
}

func channelSend(c _channel, val unsafe.Pointer) {
	c.cond.L.Lock()
	defer c.cond.L.Unlock()

	if *c.closed {
		// Channel is closed, cannot send
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
			panic(plainError("send on closed channel"))
		}
		c.cond.Wait()
		nextWriteIndex = (*c.writeIndex + 1) % actualCap
	}

	// Send the value
	memcpy(unsafe.Add(c.buffer, uintptr(*c.writeIndex)*c.chanType.channel.elementType.size), val, c.chanType.channel.elementType.size)

	// Update write index
	*c.writeIndex = nextWriteIndex

	// Buffer is now full if writeIndex == readIndex
	*c.full = *c.writeIndex == *c.readIndex

	// Signal any goroutines waiting to receive
	c.cond.Signal()
}

func channelReceive(c _channel, result unsafe.Pointer, block bool) (ok bool) {
	c.cond.L.Lock()
	defer c.cond.L.Unlock()

	if c.readIndex == c.writeIndex {
		if *c.closed || !block {
			// Receive the zero value immediately
			memset(result, 0, c.chanType.channel.elementType.size)
			return false
		}
	}

	// Block the current _goroutine until there is a value to receive
	for ; *c.readIndex == *c.writeIndex && !*c.full; c.cond.Wait() {
	}

	// Receive the value
	memcpy(result, unsafe.Add(c.buffer, uintptr(*c.readIndex)*c.chanType.channel.elementType.size), c.chanType.channel.elementType.size)

	// The channel needs to be able to store at least one item
	actualCap := c.capacity
	if actualCap == 0 {
		actualCap = 1
	}

	// Advance the read index, wrapping around if necessary
	*c.readIndex = (*c.readIndex + 1) % actualCap

	*c.full = false

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
