package http

import (
	"errors"
	"io"
)

// DefaultClient is the package-level Client used by the shortcut
// functions Get, Head, Post, and PostBytes.
//
// It uses default settings: net.Dial as the dialer, a 30-second idle
// TTL on the connection pool, and one retry on stale pooled conns for
// idempotent requests.
//
// For embedded use, you can replace it at startup:
//
//	http.DefaultClient = &http.Client{
//	    Dial:    myCustomDialer,
//	    IdleTTL: 60 * time.Second,
//	}
//
// or simply construct your own Client and use its methods directly,
// which is preferable in hot paths because the shortcuts allocate.
var DefaultClient = &Client{}

// defaultRespBufSize is the size of the response buffer the shortcut
// functions allocate. Big enough for typical API response headers
// (a few hundred bytes is normal) plus prefetch room for small bodies.
//
// If your server emits unusually large headers (lots of cookies, big
// CORS preambles), bump this. Or: don't use the shortcuts and call
// Client.Do with your own buffer.
const defaultRespBufSize = 2048

// Get issues a GET to the given URL using DefaultClient.
//
// Unlike Client.Do, this function ALLOCATES — both the response buffer
// (a fresh 2 KB []byte per call) and the *Response. Use it for
// occasional fetches where the convenience matters more than the
// allocations: boot-time config, ad-hoc telemetry posts, scripts.
//
// For hot loops or memory-critical paths, use a Client value and call
// Do or DoBuf directly with caller-supplied storage.
//
// The caller MUST call resp.Body.Close() when done to release the
// underlying connection back to the pool (or close it). The response
// buffer the function allocated lives as long as the Response — it's
// referenced indirectly through the Header strings.
func Get(rawURL string) (*Response, error) {
	return DefaultClient.GetURL(rawURL)
}

// Head issues a HEAD request. Same allocation tradeoff as Get.
func Head(rawURL string) (*Response, error) {
	return DefaultClient.HeadURL(rawURL)
}

// Post issues a POST with body of the given content type. body may be
// nil for a zero-length post.
//
// If body is non-nil, the request uses chunked transfer encoding
// (ContentLength = -1). If you know the length in advance, use
// PostBytes instead — it sends a single Content-Length-framed request,
// which is friendlier to picky servers.
func Post(rawURL, contentType string, body io.Reader) (*Response, error) {
	return DefaultClient.PostURL(rawURL, contentType, body)
}

// PostBytes is like Post but takes a []byte. It sends the body with
// an explicit Content-Length header (no chunking) and avoids needing
// the caller to construct an io.Reader.
//
// The body slice is referenced (not copied) until the request
// completes — caller must not mutate it during the call.
func PostBytes(rawURL, contentType string, body []byte) (*Response, error) {
	return DefaultClient.PostBytesURL(rawURL, contentType, body)
}

// GetURL is the URL-taking variant of Client.Get. Allocates a response
// buffer and Response struct on the caller's behalf — see the package
// docs on Get for the allocation discussion.
func (c *Client) GetURL(rawURL string) (*Response, error) {
	scheme, host, path, err := parseURL(rawURL)
	if err != nil {
		return nil, err
	}
	if scheme != 0 {
		// HTTPS not implemented yet. When it is, this dispatches to
		// the TLS-wrapping dialer.
		return nil, errors.New("http: https not yet supported")
	}

	resp := &Response{}
	buf := make([]byte, defaultRespBufSize)

	var req Request
	req.Method = MethodGet
	req.Host = host
	req.Path = path

	if err := c.Do(&req, buf, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// HeadURL is the URL-taking variant of a HEAD request.
//
// HEAD responses have no body by definition (RFC 7230 §3.3.3 rule 1).
// Per spec, the response is framed as if it had a body — Content-Length
// or Transfer-Encoding may be present and indicate what would have
// been sent — but no body bytes follow. The codec handles this by
// returning EOF on the first Body.Read regardless of declared framing.
func (c *Client) HeadURL(rawURL string) (*Response, error) {
	scheme, host, path, err := parseURL(rawURL)
	if err != nil {
		return nil, err
	}
	if scheme != 0 {
		return nil, errors.New("http: https not yet supported")
	}

	resp := &Response{}
	buf := make([]byte, defaultRespBufSize)

	var req Request
	req.Method = MethodHead
	req.Host = host
	req.Path = path

	if err := c.Do(&req, buf, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// PostURL posts an io.Reader body with chunked transfer encoding.
func (c *Client) PostURL(rawURL, contentType string, body io.Reader) (*Response, error) {
	scheme, host, path, err := parseURL(rawURL)
	if err != nil {
		return nil, err
	}
	if scheme != 0 {
		return nil, errors.New("http: https not yet supported")
	}

	resp := &Response{}
	buf := make([]byte, defaultRespBufSize)

	var req Request
	req.Method = MethodPost
	req.Host = host
	req.Path = path
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if body != nil {
		req.Body = body
		req.ContentLength = -1 // chunked
	}

	if err := c.Do(&req, buf, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// PostBytesURL posts a []byte body with explicit Content-Length.
func (c *Client) PostBytesURL(rawURL, contentType string, body []byte) (*Response, error) {
	scheme, host, path, err := parseURL(rawURL)
	if err != nil {
		return nil, err
	}
	if scheme != 0 {
		return nil, errors.New("http: https not yet supported")
	}

	resp := &Response{}
	buf := make([]byte, defaultRespBufSize)
	br := NewBytesReader(body)

	var req Request
	req.Method = MethodPost
	req.Host = host
	req.Path = path
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	req.Body = &br
	req.ContentLength = int64(len(body))

	if err := c.Do(&req, buf, resp); err != nil {
		return nil, err
	}
	return resp, nil
}
