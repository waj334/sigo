package ringbuffer

import "errors"

type RingBuffer struct {
	buffer []byte
	begin  uint8
	end    uint8
	full   bool
}

var (
	errBufferIsEmpty = errors.New("buffer is empty")
	errBufferIsFull  = errors.New("buffer is full")
)

const (
	defaultBufferSz = 256
)

func New(sz uintptr) RingBuffer {
	if sz == 0 {
		sz = defaultBufferSz
	}

	arr := make([]byte, sz)
	buf := RingBuffer{
		buffer: arr,
		begin:  0,
		end:    0,
		full:   false,
	}

	buf.buffer = arr

	return buf
}

func (r *RingBuffer) Read(p []byte) (n int, err error) {
	if r.end > r.begin {
		// Copy between start and end
		n = copy(p, r.buffer[r.begin:r.end])
	} else {
		// Copy to end of buffer first
		n = copy(p, r.buffer[r.begin:])
		if n < len(p) {
			// Copy from start to end
			n += copy(p, r.buffer[:r.end])
		}
	}

	if n > 0 {
		// Note: This takes advantage of integer overflow to wrap around
		r.begin += uint8(n)
		r.full = false
	}
	return
}

func (r *RingBuffer) Write(p []byte) (n int, err error) {
	for n < len(p) {
		if r.end > r.begin {
			n += copy(r.buffer[r.end:], p)
		} else {
			n += copy(r.buffer[r.end:r.begin], p)
		}

		if r.end == r.begin {
			// The buffer is full
			r.full = true
			break
		}

		n += copy(r.buffer[r.end:], p)
		r.end += uint8(n)
	}
	return
}

func (r *RingBuffer) WriteString(str string) (n int, err error) {
	for _, b := range str {
		if err = r.WriteByte(byte(b)); err != nil {
			return n, err
		}
		n++
	}
	return
}

func (r *RingBuffer) ReadByte() (byte, error) {
	if !r.full && r.end == r.begin {
		return 0, errBufferIsEmpty
	}

	// Get the current byte from the buffer
	b := r.buffer[r.begin]

	// Advance the begin iterator to the next byte
	r.begin++

	// The buffer would no longer be full
	r.full = false

	// Return
	return b, nil
}

func (r *RingBuffer) WriteByte(b byte) error {
	if r.full {
		return errBufferIsFull
	}

	// Set the current byte
	r.buffer[r.end] = b

	// Advance the end iterator to the next byte
	r.end++

	// Check if the next byte is the begin iterator
	if r.end == r.begin {
		// The buffer is full
		r.full = true
	}

	return nil
}

func (r *RingBuffer) Len() int {
	if r.full {
		return len(r.buffer)
	} else if r.end > r.begin {
		return int(r.end - r.begin)
	} else {
		return (len(r.buffer) - int(r.begin)) + int(r.end)
	}
}
