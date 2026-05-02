package http

import (
	"errors"
	"io"
	"time"

	"net"
)

// DialFunc is the signature for a custom dialer. The default uses
// net.Dial; supply your own to layer TLS, log, or rate-limit.
type DialFunc func(network, addr string) (net.Conn, error)

// Client is the HTTP client. A zero Client is usable but uses no
// connection pooling and the default dialer.
type Client struct {
	// Dial is the function used to create new connections.
	// If nil, net.Dial is used.
	Dial DialFunc

	// IdleTTL is the maximum time a pooled idle conn may sit before
	// it's discarded on next access. Zero means defaultIdleTTL (30s).
	// Negative means no expiry (not recommended — eventually you'll
	// race with a server-side timeout).
	IdleTTL time.Duration

	// MaxRetries is the number of times Do will retry a request that
	// fails on a stale pooled connection. Only idempotent methods are
	// retried. Default is 1, which is sufficient — if a freshly dialed
	// conn also fails, the issue is not staleness.
	MaxRetries int

	pool     connPool
	poolInit bool // lazily initialized on first Do
}

// initPool sets up the pool's idle-TTL on first use. Done lazily so a
// zero Client is usable without an explicit constructor.
func (c *Client) initPool() {
	if c.poolInit {
		return
	}
	switch {
	case c.IdleTTL == 0:
		c.pool.idleTTL = defaultIdleTTL
	case c.IdleTTL < 0:
		c.pool.idleTTL = 0 // sentinel for "never expire"
	default:
		c.pool.idleTTL = c.IdleTTL
	}
	c.poolInit = true
}

// Do sends req and populates resp. respBuf is used as scratch for the
// request line and headers, and as the parse buffer for the response
// status line and headers. Its size caps the maximum size of the
// response header block.
//
// On success, resp is populated and the caller must eventually call
// resp.Body.Close(). On error, resp is left in an indeterminate state
// (don't read from it) and any conn obtained from the pool has been
// discarded.
//
// respBuf must remain live and unmodified until the caller is done
// with resp — header strings on resp are slices into it.
//
// Typical sizing for respBuf is 1024-2048 bytes. 1024 fits a typical
// JSON API response header section with prefetch room for small bodies;
// 2048 covers fatter headers (cookies, CORS, telemetry).
func (c *Client) Do(req *Request, respBuf []byte, resp *Response) error {
	if len(respBuf) < 128 {
		return ErrBufferTooSmall
	}
	c.initPool()

	maxAttempts := c.MaxRetries + 1
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		isRetry := attempt > 0

		// Get a connection — pooled if available, freshly dialed
		// otherwise. On retry we always dial fresh, since the prior
		// attempt's failure suggests pooled conns are unreliable.
		conn, slot, key, err := c.acquireConn(req, isRetry)
		if err != nil {
			return err
		}

		err = c.attemptRoundTrip(req, respBuf, resp, conn, slot, key)
		if err == nil {
			return nil
		}

		lastErr = err

		// Decide whether to retry. We only retry if:
		//  1. The request method is idempotent.
		//  2. The conn we used was from the pool (slot >= 0). A failure
		//     on a freshly dialed conn isn't going to be fixed by
		//     dialing another fresh conn.
		if !req.isIdempotent() || slot < 0 {
			return err
		}
		// Loop and try again with a fresh conn.
	}
	return lastErr
}

// acquireConn returns a conn for req — pooled if possible, else freshly
// dialed. If forceFresh is true, the pool is bypassed.
func (c *Client) acquireConn(req *Request, forceFresh bool) (net.Conn, int, connKey, error) {
	key := connKey{host: req.Host, scheme: 0} // TODO: scheme=1 for HTTPS once TLS lands

	if !forceFresh {
		if conn, slot := c.pool.get(key, nowMs()); conn != nil {
			return conn, slot, key, nil
		}
	}

	dial := c.Dial
	if dial == nil {
		dial = net.Dial
	}
	conn, err := dial("tcp", req.Host)
	if err != nil {
		return nil, -1, key, err
	}
	return conn, -1, key, nil
}

// attemptRoundTrip writes the request, reads the response headers, and
// initializes resp.Body. On any error, the conn is closed (or discarded
// from the pool) — the caller's retry logic decides whether to dial again.
func (c *Client) attemptRoundTrip(
	req *Request, respBuf []byte, resp *Response,
	conn net.Conn, slot int, key connKey,
) error {
	if err := writeRequest(conn, req, respBuf); err != nil {
		c.pool.discard(conn, slot)
		return err
	}
	if req.Body != nil && req.ContentLength != 0 {
		if err := writeBody(conn, req, respBuf); err != nil {
			c.pool.discard(conn, slot)
			return err
		}
	}

	headerEnd, prefetch, err := readHeaders(conn, respBuf)
	if err != nil {
		c.pool.discard(conn, slot)
		return err
	}

	if err := parseResponse(respBuf[:headerEnd], resp); err != nil {
		c.pool.discard(conn, slot)
		return err
	}

	// Initialize the body reader with the prefetch and framing info.
	resp.Body = BodyReader{
		conn:     conn,
		prefetch: prefetch,
		client:   c,
		key:      key,
		slot:     slot,
	}

	// RFC 7230 §3.3.3: certain responses MUST NOT have a body, even if
	// Content-Length or Transfer-Encoding suggests otherwise.
	//   - Any 1xx (informational) response.
	//   - 204 No Content.
	//   - 304 Not Modified.
	//   - Any response to a HEAD request.
	// We override the framing to "no body" in these cases. The
	// underlying conn is still usable for further requests because
	// the next bytes on the wire are guaranteed to be a fresh response.
	noBody := req.Method == MethodHead ||
		resp.StatusCode == 204 ||
		resp.StatusCode == 304 ||
		(resp.StatusCode >= 100 && resp.StatusCode < 200)

	if noBody {
		resp.Body.remaining = 0
		// Don't overwrite resp.ContentLength — the spec says the value
		// the server sent is meaningful (e.g. HEAD's Content-Length
		// reports what a GET would return). Leaving it lets the caller
		// inspect it.
	} else {
		switch resp.ContentLength {
		case -1:
			resp.Body.remaining = -1
			resp.Body.chunkState = chunkStateNeedSize
		case -2:
			resp.Body.remaining = -2
		default:
			resp.Body.remaining = resp.ContentLength
		}
	}

	return nil
}

// Get is a convenience wrapper for issuing a GET. The caller still
// supplies the response buffer and Response struct so the call stays
// allocation-free.
func (c *Client) Get(host, path string, respBuf []byte, resp *Response) error {
	var req Request
	req.Method = MethodGet
	req.Host = host
	req.Path = path
	return c.Do(&req, respBuf, resp)
}

// CloseIdleConnections closes all idle conns in the pool. Conns
// currently in-use are unaffected and will be closed (or returned to
// the pool, which is now empty) by their owners as usual.
func (c *Client) CloseIdleConnections() {
	c.initPool()
	c.pool.closeAll()
}

// BytesReader is a no-allocation io.Reader over a fixed []byte.
//
// Use as a Request.Body for small fixed payloads:
//
//	body := http.NewBytesReader(payload)
//	req.Body = &body
//	req.ContentLength = int64(len(payload))
//
// Unlike bytes.Reader from the standard library, this type is meant
// to be allocated on the goroutine stack and used by pointer, so the
// caller controls the storage.
type BytesReader struct {
	data []byte
	pos  int
}

// NewBytesReader returns a BytesReader ready to read from data.
// Returned by value so the caller can keep it on the stack.
func NewBytesReader(data []byte) BytesReader {
	return BytesReader{data: data}
}

// Read implements io.Reader.
func (r *BytesReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// Reset rewinds the reader so it can be used again. Useful when a
// retry needs to re-send the same body.
func (r *BytesReader) Reset() { r.pos = 0 }

// (sanity checks at compile time)
var (
	_ io.Reader     = (*BytesReader)(nil)
	_ io.ReadCloser = (*BodyReader)(nil)
	_               = errors.New // keep errors imported; used in body.go and elsewhere
)
