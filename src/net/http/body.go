package http

import (
	"errors"
	"io"

	"net"
)

// BodyReader streams a response body. It is a concrete type rather
// than an interface so the hot path (Read on a length-known body)
// inlines and avoids an itab allocation. It still satisfies io.ReadCloser.
//
// Three framing modes are supported, selected automatically from the
// response headers:
//
//  1. Identity with Content-Length: read exactly N bytes.
//  2. Chunked transfer-encoding: read each chunk's hex size, then its
//     bytes, then CRLF, repeat until size 0.
//  3. Read until close: no Content-Length, no chunked encoding.
//
// In all three modes, Read may return data from the prefetch buffer
// (bytes that came in the same TCP read as the headers) before
// touching the underlying conn.
type BodyReader struct {
	conn     net.Conn
	prefetch []byte // unread bytes already in the response buffer

	// remaining is the number of identity-mode bytes left to read,
	// or one of the negative sentinels:
	//   -1 : chunked encoding (chunk* fields are live)
	//   -2 : read until conn close
	//    0 : body fully consumed
	remaining int64

	// Chunked-decoder state. Used only when remaining == -1.
	chunkRem   int64 // bytes left in the current chunk's data
	chunkState uint8 // see chunk* constants below
	chunkLine  [20]byte
	chunkLineN uint8

	// pool/slot/client are wired up by Client.Do so Close can return
	// the conn or discard it. They're zero-valued for Bodies returned
	// from a hypothetical lower-level API that doesn't pool — Close
	// closes the conn directly in that case.
	client *Client
	key    connKey
	slot   int  // -1 if conn was freshly dialed (not pooled)
	closed bool // set by Close to make Close idempotent
	bad    bool // set on protocol error so Close discards instead of pools
}

// Chunked decoder states.
const (
	chunkStateNeedSize  uint8 = iota // collecting hex digits and ext into chunkLine
	chunkStateInChunk                // reading chunkRem bytes of data
	chunkStateNeedCRLF1              // expecting '\r' after chunk data
	chunkStateNeedCRLF2              // expecting '\n' after the '\r'
	chunkStateTrailers               // reading optional trailer headers (we discard them)
	chunkStateDone                   // last chunk seen, full message complete
)

// Read implements io.Reader.
func (b *BodyReader) Read(p []byte) (int, error) {
	if b.closed {
		return 0, ErrBodyClosed
	}
	if len(p) == 0 {
		return 0, nil
	}

	switch {
	case b.remaining == 0:
		return 0, io.EOF
	case b.remaining > 0:
		return b.readIdentity(p)
	case b.remaining == -1:
		return b.readChunked(p)
	case b.remaining == -2:
		return b.readUntilClose(p)
	}
	return 0, errors.New("http: bad body state")
}

// readIdentity reads up to len(p) bytes from prefetch + conn,
// capped at b.remaining (the Content-Length residue).
func (b *BodyReader) readIdentity(p []byte) (int, error) {
	want := int64(len(p))
	if want > b.remaining {
		want = b.remaining
	}

	// Drain prefetch first.
	if len(b.prefetch) > 0 {
		n := copy(p[:want], b.prefetch)
		b.prefetch = b.prefetch[n:]
		b.remaining -= int64(n)
		if b.remaining == 0 {
			// Body complete. Don't touch the conn; the next Read
			// returns EOF, which is the contract.
		}
		return n, nil
	}

	n, err := b.conn.Read(p[:want])
	if n > 0 {
		b.remaining -= int64(n)
	}
	if err != nil {
		if err == io.EOF && b.remaining > 0 {
			// Server closed before sending the full Content-Length.
			b.bad = true
			return n, ErrUnexpectedEOF
		}
		return n, err
	}
	return n, nil
}

// readUntilClose reads bytes until the conn returns EOF.
// EOF is the *expected* terminator here, not an error.
func (b *BodyReader) readUntilClose(p []byte) (int, error) {
	if len(b.prefetch) > 0 {
		n := copy(p, b.prefetch)
		b.prefetch = b.prefetch[n:]
		return n, nil
	}
	n, err := b.conn.Read(p)
	if err == io.EOF {
		b.remaining = 0
		// Mark bad so Close discards: a connection that's been read
		// to EOF is by definition not reusable.
		b.bad = true
	}
	return n, err
}

// readChunked drives the chunked-decoder state machine. It produces at
// most len(p) bytes per call but may produce fewer if a chunk boundary
// or buffer-fill stop comes first.
//
// The loop invariant is: each iteration either advances the state
// machine or produces output. We never spin without making progress.
func (b *BodyReader) readChunked(p []byte) (int, error) {
	written := 0
	for written < len(p) {
		switch b.chunkState {
		case chunkStateNeedSize:
			done, err := b.readChunkSizeLine()
			if err != nil {
				return written, err
			}
			if !done {
				if written > 0 {
					return written, nil
				}
				// We need more bytes from the conn but produced
				// nothing — propagate the underlying error/EOF if any
				// (readChunkSizeLine already returned them).
				return 0, nil
			}
			// chunkRem and chunkState now set by readChunkSizeLine.

		case chunkStateInChunk:
			n, err := b.readChunkData(p[written:])
			written += n
			if err != nil {
				return written, err
			}
			// readChunkData transitions to chunkStateNeedCRLF1 when
			// chunkRem hits zero. We keep looping in that case to
			// consume the CRLF without returning.

		case chunkStateNeedCRLF1, chunkStateNeedCRLF2:
			done, err := b.readChunkTrailingCRLF()
			if err != nil {
				return written, err
			}
			if !done {
				if written > 0 {
					return written, nil
				}
				return 0, nil
			}

		case chunkStateTrailers:
			done, err := b.readTrailers()
			if err != nil {
				return written, err
			}
			if !done {
				if written > 0 {
					return written, nil
				}
				return 0, nil
			}

		case chunkStateDone:
			b.remaining = 0
			if written > 0 {
				return written, nil
			}
			return 0, io.EOF
		}
	}
	return written, nil
}

// readChunkSizeLine accumulates bytes into b.chunkLine until it sees
// CRLF, then parses the hex size, then transitions state.
//
// Returns done=true when the size line is fully parsed (regardless of
// whether the size was zero or nonzero — chunkState reflects the
// transition).
func (b *BodyReader) readChunkSizeLine() (done bool, err error) {
	for {
		// Get one byte from prefetch or conn.
		c, ok, err := b.readByte()
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}

		// Strip "\r" — we'll consume "\n" as the line terminator.
		// This is lenient by design: some servers omit the CR.
		if c == '\r' {
			continue
		}
		if c == '\n' {
			// Line complete. Truncate at the first ';' to drop the
			// chunk-extension, which we ignore entirely.
			line := b.chunkLine[:b.chunkLineN]
			if i := indexByte(line, ';'); i >= 0 {
				line = line[:i]
			}
			// Strip trailing whitespace (some servers add it before
			// the CRLF for no good reason).
			for len(line) > 0 && (line[len(line)-1] == ' ' || line[len(line)-1] == '\t') {
				line = line[:len(line)-1]
			}
			n, ok := parseHexUint(line)
			if !ok {
				b.bad = true
				return false, ErrBadChunkSize
			}
			b.chunkLineN = 0
			if n == 0 {
				// Last chunk. Move on to trailers (which we discard).
				b.chunkState = chunkStateTrailers
			} else {
				b.chunkRem = int64(n)
				b.chunkState = chunkStateInChunk
			}
			return true, nil
		}

		if int(b.chunkLineN) == len(b.chunkLine) {
			// Chunk-size line absurdly long — almost certainly garbage.
			b.bad = true
			return false, ErrBadChunkSize
		}
		b.chunkLine[b.chunkLineN] = c
		b.chunkLineN++
	}
}

// readChunkData reads up to min(len(p), chunkRem) bytes from the body
// stream into p. When chunkRem hits zero, it transitions to
// chunkStateNeedCRLF1.
func (b *BodyReader) readChunkData(p []byte) (int, error) {
	if b.chunkRem == 0 {
		b.chunkState = chunkStateNeedCRLF1
		return 0, nil
	}
	want := int64(len(p))
	if want > b.chunkRem {
		want = b.chunkRem
	}

	// Drain prefetch before touching the conn.
	if len(b.prefetch) > 0 {
		n := copy(p[:want], b.prefetch)
		b.prefetch = b.prefetch[n:]
		b.chunkRem -= int64(n)
		if b.chunkRem == 0 {
			b.chunkState = chunkStateNeedCRLF1
		}
		return n, nil
	}

	n, err := b.conn.Read(p[:want])
	if n > 0 {
		b.chunkRem -= int64(n)
		if b.chunkRem == 0 {
			b.chunkState = chunkStateNeedCRLF1
		}
	}
	if err != nil {
		if err == io.EOF {
			b.bad = true
			return n, ErrUnexpectedEOF
		}
		return n, err
	}
	return n, nil
}

// readChunkTrailingCRLF consumes the "\r\n" after a chunk's data.
func (b *BodyReader) readChunkTrailingCRLF() (done bool, err error) {
	for {
		c, ok, err := b.readByte()
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
		switch b.chunkState {
		case chunkStateNeedCRLF1:
			if c == '\n' {
				// Lenient: bare LF.
				b.chunkState = chunkStateNeedSize
				return true, nil
			}
			if c != '\r' {
				b.bad = true
				return false, ErrChunkedAfterRead
			}
			b.chunkState = chunkStateNeedCRLF2
		case chunkStateNeedCRLF2:
			if c != '\n' {
				b.bad = true
				return false, ErrChunkedAfterRead
			}
			b.chunkState = chunkStateNeedSize
			return true, nil
		}
	}
}

// readTrailers consumes the optional trailer-part after the last chunk.
// We don't expose trailers to the caller — they're rare in practice and
// supporting them would balloon the API.
//
// The trailer-part is a sequence of header-fields terminated by an
// empty line, just like the main header block.
func (b *BodyReader) readTrailers() (done bool, err error) {
	// Reuse chunkLine as scratch for one trailer line at a time.
	for {
		c, ok, err := b.readByte()
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
		if c == '\r' {
			continue
		}
		if c == '\n' {
			if b.chunkLineN == 0 {
				// Empty line — end of trailers.
				b.chunkState = chunkStateDone
				return true, nil
			}
			// End of one trailer line; discard and start over.
			b.chunkLineN = 0
			continue
		}
		// Non-terminator byte. Just count it; we don't care what it is.
		// We do cap the line length to catch garbage streams.
		if int(b.chunkLineN) == len(b.chunkLine) {
			// Don't error here — a real trailer header could legitimately
			// be longer than 20 bytes. Just stop accumulating; we'll
			// still consume bytes and look for the line terminator.
			continue
		}
		b.chunkLine[b.chunkLineN] = c
		b.chunkLineN++
	}
}

// readByte returns one byte from prefetch or the conn.
//
// Returns ok=false (with no error) when the conn returns 0 bytes
// without an error — caller should propagate as "no progress this
// call". Returns ok=true on success. Returns an error on EOF or
// transport failure.
func (b *BodyReader) readByte() (c byte, ok bool, err error) {
	if len(b.prefetch) > 0 {
		c = b.prefetch[0]
		b.prefetch = b.prefetch[1:]
		return c, true, nil
	}
	var buf [1]byte
	n, err := b.conn.Read(buf[:])
	if n > 0 {
		return buf[0], true, nil
	}
	if err != nil {
		return 0, false, err
	}
	return 0, false, nil
}

// Close releases the conn back to the pool (if reusable) or closes it
// (if poisoned by a protocol error or partial body read).
//
// Close is idempotent — calling it more than once is safe.
//
// After Close, Read returns ErrBodyClosed.
func (b *BodyReader) Close() error {
	if b.closed {
		return nil
	}
	b.closed = true

	if b.client == nil || b.conn == nil {
		// No client wired up (lower-level usage). Just close the conn.
		if b.conn != nil {
			return b.conn.Close()
		}
		return nil
	}

	// If the body is bad or only partially consumed, the conn is not
	// reusable — discard it. Otherwise return to the pool.
	if b.bad || b.remaining != 0 || b.chunkState != chunkStateDone && b.remaining == -1 {
		b.client.pool.discard(b.conn, b.slot)
		return nil
	}

	b.client.pool.release(b.conn, b.key, b.slot, nowMs())
	return nil
}
