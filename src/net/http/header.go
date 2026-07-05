package http

import (
	"errors"
	"io"
	"unsafe"

	"net"
)

// Method constants. Defined as package-level strings so callers can
// reference them without writing string literals (which become per-call
// allocations on some compilers if not interned).
const (
	MethodGet     = "GET"
	MethodHead    = "HEAD"
	MethodPost    = "POST"
	MethodPut     = "PUT"
	MethodDelete  = "DELETE"
	MethodPatch   = "PATCH"
	MethodOptions = "OPTIONS"
)

// maxHeaders caps the number of header fields we'll parse from a response.
// 24 covers every real-world API I've seen; servers that emit more are
// almost always misbehaving and rejecting them surfaces the bug rather
// than silently truncating.
const maxHeaders = 24

// Header is a bounded, allocation-free header set.
//
// Backing storage is an inline array — Header has no pointer fields,
// which means a Header value lives entirely on the goroutine stack (or
// inside its containing Request/Response without an extra heap object).
//
// Names and values are strings. On a Request they're caller-supplied
// (typically string literals or fields of caller-owned structs). On a
// Response they alias the response buffer passed to Client.Do — the
// caller must keep that buffer live and unmodified while using the
// Response.
type Header struct {
	pairs [maxHeaders]headerPair
	n     uint8
}

type headerPair struct {
	name, value string
}

// Get returns the first value associated with name, case-insensitive.
// Returns "" if the header is not present.
func (h *Header) Get(name string) string {
	for i := uint8(0); i < h.n; i++ {
		if equalFoldASCII(h.pairs[i].name, name) {
			return h.pairs[i].value
		}
	}
	return ""
}

// Set assigns value to name, replacing any existing value.
// Returns ErrHeaderFull if the table is full and name is not already present.
func (h *Header) Set(name, value string) error {
	for i := uint8(0); i < h.n; i++ {
		if equalFoldASCII(h.pairs[i].name, name) {
			h.pairs[i].value = value
			return nil
		}
	}
	if h.n == maxHeaders {
		return ErrHeaderFull
	}
	h.pairs[h.n] = headerPair{name, value}
	h.n++
	return nil
}

// Del removes all entries matching name.
func (h *Header) Del(name string) {
	w := uint8(0)
	for r := uint8(0); r < h.n; r++ {
		if !equalFoldASCII(h.pairs[r].name, name) {
			if w != r {
				h.pairs[w] = h.pairs[r]
			}
			w++
		}
	}
	h.n = w
}

// Len reports the number of header entries.
func (h *Header) Len() int { return int(h.n) }

// At returns the i'th header pair. Panics if i is out of range.
// Useful for iteration without exposing the internal array.
func (h *Header) At(i int) (name, value string) {
	return h.pairs[i].name, h.pairs[i].value
}

// add appends without checking for duplicates. Used internally by the
// parser, which by construction never adds the same name twice from a
// single response unless the server sent it twice (which is legal for
// some headers like Set-Cookie — but on response we treat every header
// as single-valued and let Get return the first match).
func (h *Header) add(name, value string) error {
	if h.n == maxHeaders {
		return ErrTooManyHeaders
	}
	h.pairs[h.n] = headerPair{name, value}
	h.n++
	return nil
}

// reset clears the header set without touching the backing array.
// The strings in the array are abandoned (no work to do — they're
// either literals or slices into a buffer the caller owns).
func (h *Header) reset() { h.n = 0 }

// Request is an HTTP request. All string fields are borrowed references —
// the caller must keep them live and unmodified until Client.Do returns.
//
// For zero-allocation usage, allocate Request on the goroutine stack:
//
//	var req Request
//	req.Method = MethodGet
//	req.Host = "10.0.0.1:8080"
//	req.Path = "/api/v1/status"
//	req.Header.Set("Accept", "application/json")
//	err := client.Do(&req, buf[:], &resp)
type Request struct {
	// Method is the HTTP method. Use one of the Method* constants.
	Method string

	// Host is exactly the string passed to net.Dial — typically "ip:port".
	// Also used as the value of the Host: header on the wire.
	Host string

	// Path is the request-target, including any query string. The codec
	// emits it verbatim, so the caller is responsible for encoding.
	// Should always begin with "/".
	Path string

	// Header is the set of request headers. Host, Content-Length, and
	// Transfer-Encoding are managed by the codec — don't set them here.
	Header Header

	// Body is the request body. nil means no body.
	// For small fixed bodies, BytesReader avoids allocations.
	Body io.Reader

	// ContentLength governs how the body is framed:
	//  - 0 with Body == nil: no body
	//  - >0: emitted as Content-Length: N, body must produce exactly N bytes
	//  - -1: chunked transfer-encoding (use when length is unknown)
	ContentLength int64
}

// isIdempotent reports whether r's method may be retried after a transport
// error per RFC 7231 §4.2.2. This drives the retry-on-stale-pool-conn
// path in Client.Do — non-idempotent methods (POST, PATCH) are surfaced
// to the caller on transport error, because retrying could double-apply
// a side effect.
func (r *Request) isIdempotent() bool {
	switch r.Method {
	case MethodGet, MethodHead, MethodPut, MethodDelete, MethodOptions:
		return true
	}
	return false
}

// Response is the HTTP response from a server. All strings except
// possibly Body's underlying bytes alias the response buffer supplied
// to Client.DoBuf. The caller must keep that buffer live and unmodified
// for as long as the Response is in use.
type Response struct {
	// StatusCode is the parsed numeric status (e.g. 200, 404).
	StatusCode int

	// Status is the full status text including the code, e.g. "200 OK".
	// Slices into the response buffer.
	Status string

	// Header is the parsed response headers. Names and values slice into
	// the response buffer.
	Header Header

	// ContentLength is the parsed Content-Length, or -1 for chunked
	// transfer-encoding, or -2 if neither was set (read until close).
	ContentLength int64

	// Body streams the response body. The caller MUST call Body.Close()
	// when done — that's what returns the conn to the pool.
	Body BodyReader
}

// readHeaders reads from conn into buf until "\r\n\r\n" is found.
//
// Returns the offset where headers end (one past the final \n) and any
// bytes that were read past the headers (the body-prefetch). For small
// responses the prefetch contains the entire body, and the codec can
// return without ever calling Read on conn again.
//
// If buf fills before "\r\n\r\n" is seen, returns ErrHeadersTooLarge —
// the caller should bump the buffer size or stop talking to that server.
func readHeaders(conn net.Conn, buf []byte) (headerEnd int, prefetch []byte, err error) {
	n := 0
	for {
		if n == len(buf) {
			return 0, nil, ErrHeadersTooLarge
		}
		m, rerr := conn.Read(buf[n:])
		if m > 0 {
			// Look for the sentinel including up to 3 bytes from before
			// this read, in case "\r\n\r\n" straddles the boundary.
			scanStart := n - 3
			if scanStart < 0 {
				scanStart = 0
			}
			region := buf[scanStart : n+m]
			if idx := findDoubleCRLF(region); idx >= 0 {
				headerEnd = scanStart + idx + 4
				prefetch = buf[headerEnd : n+m]
				return headerEnd, prefetch, nil
			}
			n += m
		}
		if rerr != nil {
			// EOF or transport error before we found the sentinel.
			if rerr == io.EOF {
				return 0, nil, ErrUnexpectedEOF
			}
			return 0, nil, rerr
		}
	}
}

// parseResponse parses the status line and headers in buf (which must
// end exactly at the position returned by readHeaders' headerEnd, i.e.
// include the terminating "\r\n\r\n"). It populates resp.StatusCode,
// resp.Status, and resp.Header.
//
// All resulting strings alias buf.
func parseResponse(buf []byte, resp *Response) error {
	resp.Header.reset()

	// Status line: "HTTP/1.x SSS Reason\r\n"
	lineEnd := indexCRLF(buf)
	if lineEnd < 0 {
		return ErrMalformedResponse
	}
	line := buf[:lineEnd]

	// "HTTP/" prefix and version. We accept HTTP/1.0 and HTTP/1.1.
	// HTTP/0.9 has no headers at all and isn't in scope; HTTP/2 won't
	// arrive over a plain TCP connection from this client.
	if len(line) < 12 {
		return ErrMalformedResponse
	}
	if line[0] != 'H' || line[1] != 'T' || line[2] != 'T' || line[3] != 'P' || line[4] != '/' {
		return ErrMalformedResponse
	}
	if line[5] != '1' || line[6] != '.' || (line[7] != '0' && line[7] != '1') {
		return ErrMalformedResponse
	}
	if line[8] != ' ' {
		return ErrMalformedResponse
	}

	// Status code: 3 digits starting at line[9].
	if len(line) < 12 {
		return ErrMalformedResponse
	}
	codeBytes := line[9:12]
	code, ok := parseUint(codeBytes)
	if !ok || code < 100 || code > 599 {
		return ErrMalformedResponse
	}
	resp.StatusCode = int(code)

	// Status text starts at line[9] (includes the code) and runs to end.
	// Per RFC 7230 §3.1.2 the reason-phrase may be empty, so line might
	// be exactly 12 bytes ("HTTP/1.1 200" with no trailing space). We
	// don't require a space-then-reason.
	resp.Status = bytesToString(line[9:])

	// Headers.
	pos := lineEnd + 2
	for {
		end := indexCRLF(buf[pos:])
		if end < 0 {
			return ErrMalformedResponse
		}
		if end == 0 {
			// Empty line — end of headers.
			break
		}
		hdr := buf[pos : pos+end]
		colon := indexByte(hdr, ':')
		if colon < 0 {
			return ErrMalformedResponse
		}
		name := bytesToString(hdr[:colon])
		// RFC 7230 §3.2.4: no whitespace allowed before the colon.
		// Be strict — "Content -Length: 5" should be rejected.
		for i := 0; i < colon; i++ {
			c := hdr[i]
			if c == ' ' || c == '\t' {
				return ErrMalformedResponse
			}
		}
		value := bytesToString(trimOWS(hdr[colon+1:]))
		if err := resp.Header.add(name, value); err != nil {
			return err
		}
		pos += end + 2
	}

	// Resolve framing. Transfer-Encoding takes precedence over
	// Content-Length per RFC 7230 §3.3.3.
	if te := resp.Header.Get("Transfer-Encoding"); te != "" {
		if !equalFoldASCII(te, "chunked") {
			// We don't support gzip/deflate/etc. transfer codings.
			// (Content-Encoding is different and is the application's
			// responsibility.)
			return ErrMalformedResponse
		}
		resp.ContentLength = -1
	} else if cl := resp.Header.Get("Content-Length"); cl != "" {
		n, ok := parseUint64(unsafeStringToBytes(cl))
		if !ok {
			return ErrBadContentLength
		}
		// ContentLength is int64, and -1/-2 are framing sentinels. A
		// value with bit 63 set would go negative here — in the worst
		// case landing exactly on a sentinel and selecting the wrong
		// framing mode. Reject anything that doesn't fit int64.
		if int64(n) < 0 {
			return ErrBadContentLength
		}
		resp.ContentLength = int64(n)
	} else {
		// No framing — read until the server closes.
		resp.ContentLength = -2
	}

	return nil
}

// unsafeStringToBytes converts a string to []byte without copying.
// Used only inside the codec to feed a header value (which is already
// a slice of the response buffer disguised as a string) back into the
// []byte-taking parsers in bytes.go, and to feed string literals to
// bufWriter.Write.
//
// Safe because the codec never mutates the returned slice and the
// underlying bytes outlive the parse call by virtue of the caller's
// buffer-lifetime contract on Response (for header values) or the
// fact that string literals live in read-only memory (for literals).
//
// IMPORTANT: callers MUST NOT write to the returned slice. Doing so
// for a string-literal source would page-fault on most platforms; for
// a parser-buffer source it would corrupt the response.
func unsafeStringToBytes(s string) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice(unsafe.StringData(s), len(s))
}

// writeRequest formats req's request-line and headers into a single
// Write call on conn. The body is sent separately by writeBody.
//
// scratch is a working buffer (typically the response buffer, repurposed
// before any response bytes arrive). If scratch is too small to hold
// the formatted headers, we fall back to multiple Writes — correct but
// slower. The threshold is generous; a 1024-byte scratch holds typical
// requests.
func writeRequest(conn net.Conn, req *Request, scratch []byte) error {
	w := bufWriter{buf: scratch, conn: conn}

	w.WriteString(req.Method)
	w.WriteByte(' ')
	w.WriteString(req.Path)
	w.WriteString(" HTTP/1.1\r\nHost: ")
	w.WriteString(req.Host)
	w.WriteString("\r\n")

	// Caller-supplied headers. We trust the caller didn't set Host
	// (we just wrote it) or Content-Length / Transfer-Encoding (we
	// emit those based on ContentLength below). If they did, the
	// server will see duplicates — annoying but not our problem to
	// dedupe.
	for i := uint8(0); i < req.Header.n; i++ {
		w.WriteString(req.Header.pairs[i].name)
		w.WriteString(": ")
		w.WriteString(req.Header.pairs[i].value)
		w.WriteString("\r\n")
	}

	// Body framing.
	switch {
	case req.ContentLength > 0:
		w.WriteString("Content-Length: ")
		var nbuf [20]byte
		n := writeUint(nbuf[:], uint64(req.ContentLength))
		w.Write(nbuf[:n])
		w.WriteString("\r\n")
	case req.ContentLength < 0:
		w.WriteString("Transfer-Encoding: chunked\r\n")
	}

	w.WriteString("\r\n")
	return w.Flush()
}

// writeBody streams req.Body to conn using the framing implied by
// req.ContentLength.
func writeBody(conn net.Conn, req *Request, scratch []byte) error {
	if req.Body == nil || req.ContentLength == 0 {
		return nil
	}

	if req.ContentLength > 0 {
		// Identity body: just copy req.Body to conn, capped at
		// ContentLength bytes. We use scratch as the copy buffer to
		// avoid allocating one.
		return copyN(conn, req.Body, req.ContentLength, scratch)
	}

	// ContentLength < 0: chunked.
	return copyChunked(conn, req.Body, scratch)
}

// copyN reads exactly n bytes from src and writes them to dst, using
// buf as scratch. Returns an error if src ends before n bytes are read.
func copyN(dst io.Writer, src io.Reader, n int64, buf []byte) error {
	remaining := n
	for remaining > 0 {
		want := int64(len(buf))
		if want > remaining {
			want = remaining
		}
		nr, rerr := src.Read(buf[:want])
		if nr > 0 {
			if _, werr := dst.Write(buf[:nr]); werr != nil {
				return werr
			}
			remaining -= int64(nr)
		}
		if rerr != nil {
			if rerr == io.EOF && remaining == 0 {
				return nil
			}
			return rerr
		}
	}
	return nil
}

// copyChunked streams src to dst using chunked transfer-encoding.
// Each Read from src becomes one chunk on the wire.
func copyChunked(dst io.Writer, src io.Reader, buf []byte) error {
	// Reserve the head of buf for the chunk-size line. Worst case is
	// "FFFFFFFFFFFFFFFF\r\n" = 18 bytes for a uint64 chunk size, plus
	// the trailing "\r\n" after the chunk data. So we use the first
	// 18 bytes for size and the last 2 for trailer, leaving the middle
	// for data. For small bufs, we send size and data in separate
	// Writes — correct but more wire overhead.
	const sizeReserve = 20
	if len(buf) < sizeReserve+8 {
		return errors.New("http: chunked write buffer too small")
	}
	dataBuf := buf[sizeReserve : len(buf)-2]

	for {
		nr, rerr := src.Read(dataBuf)
		if nr > 0 {
			// Format size in hex right-aligned ending at sizeReserve-2,
			// followed by "\r\n". This puts the header immediately
			// before the data so we can write them with one Write.
			//
			// Hex of nr fits in at most 16 chars (uint64 max). We use
			// up to (sizeReserve-2)=18 chars of room before the data
			// start, which is more than enough.
			sizeEnd := sizeReserve - 2 // position of the \r in "\r\n"
			buf[sizeEnd] = '\r'
			buf[sizeEnd+1] = '\n'
			pos := sizeEnd
			n := nr
			if n == 0 {
				pos--
				buf[pos] = '0'
			} else {
				for n > 0 {
					pos--
					d := byte(n & 0xF)
					if d < 10 {
						buf[pos] = '0' + d
					} else {
						buf[pos] = 'a' + d - 10
					}
					n >>= 4
				}
			}
			// Append trailing CRLF after the data.
			tailStart := sizeReserve + nr
			buf[tailStart] = '\r'
			buf[tailStart+1] = '\n'

			// Single Write: <hex>\r\n<data>\r\n
			if _, werr := dst.Write(buf[pos : tailStart+2]); werr != nil {
				return werr
			}
		}
		if rerr != nil {
			if rerr == io.EOF {
				// Terminating chunk: "0\r\n\r\n"
				_, werr := dst.Write([]byte{'0', '\r', '\n', '\r', '\n'})
				return werr
			}
			return rerr
		}
	}
}

// bufWriter buffers small writes against a fixed scratch and flushes
// on overflow or on demand. Used by writeRequest to coalesce the
// request-line and headers into a single conn.Write call.
type bufWriter struct {
	buf  []byte
	n    int
	conn net.Conn
	err  error
}

func (w *bufWriter) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	for len(p) > 0 {
		room := len(w.buf) - w.n
		if room == 0 {
			if err := w.Flush(); err != nil {
				return 0, err
			}
			continue
		}
		nn := copy(w.buf[w.n:], p)
		w.n += nn
		p = p[nn:]
	}
	return 0, nil // caller doesn't use return values; we Flush at the end
}

func (w *bufWriter) WriteString(s string) {
	_, _ = w.Write(unsafeStringToBytes(s))
}

func (w *bufWriter) WriteByte(c byte) {
	if w.err != nil {
		return
	}
	if w.n == len(w.buf) {
		if err := w.Flush(); err != nil {
			return
		}
	}
	w.buf[w.n] = c
	w.n++
}

func (w *bufWriter) Flush() error {
	if w.err != nil {
		return w.err
	}
	if w.n == 0 {
		return nil
	}
	_, err := w.conn.Write(w.buf[:w.n])
	w.n = 0
	w.err = err
	return err
}

// (No additional helpers below — bufWriter and unsafeStringToBytes
// complete the codec's internal toolkit.)
