// Package http implements a minimal HTTP/1.1 client for sigo on lwIP.
//
// This file contains the bytewise primitives the parser and codec depend on.
// They are deliberately small, branch-light, and allocation-free. Each one is
// a hot path on every request, so they are written to compile to tight loops
// the sigo backend can vectorize or unroll if it chooses.
//
// Conventions used throughout:
//   - All functions return -1 (not 0) for "not found" so a zero return value
//     unambiguously means "found at offset 0".
//   - Inputs are []byte rather than string so the caller can pass slices of
//     a network read buffer directly without conversion.
//   - No function in this file allocates.
package http

import "unsafe"

// indexByte returns the offset of the first occurrence of c in b, or -1.
//
// Equivalent in spirit to bytes.IndexByte. Defined locally so the http
// package does not depend on the bytes package (which pulls in unicode
// tables on some builds and is overkill for ASCII-only HTTP parsing).
func indexByte(b []byte, c byte) int {
	for i := 0; i < len(b); i++ {
		if b[i] == c {
			return i
		}
	}
	return -1
}

// indexCRLF returns the offset of the first "\r\n" in b, or -1.
//
// HTTP/1.1 line terminators are always CRLF (RFC 7230 §3.5 tolerates bare LF
// but only as a robustness allowance; we don't, because servers we talk to
// in embedded contexts always emit CRLF and being strict catches framing
// bugs early).
func indexCRLF(b []byte) int {
	// Scan for '\r' and check the next byte. This is faster than a two-byte
	// pattern match because '\r' is rare in header bytes — most positions
	// reject on the first comparison.
	n := len(b) - 1
	for i := 0; i < n; i++ {
		if b[i] == '\r' && b[i+1] == '\n' {
			return i
		}
	}
	return -1
}

// findDoubleCRLF returns the offset of the first "\r\n\r\n" in b, or -1.
//
// This is the end-of-headers sentinel for HTTP/1.1. The returned offset
// points at the first '\r' of the four-byte sequence; add 4 to get the
// index where the body begins.
//
// Performance note: a typical header section is 200-800 bytes, called once
// per response. The naive scan is fine — moving to a Boyer-Moore variant
// or word-at-a-time SIMD would help only on adversarially long headers,
// which we reject earlier via the buffer-full check.
func findDoubleCRLF(b []byte) int {
	// We need at least 4 bytes for the pattern. The loop bound i < len(b)-3
	// guarantees b[i+3] is in range without a per-iteration check.
	n := len(b) - 3
	for i := 0; i < n; i++ {
		// Anchor on '\r' first (rare byte) so non-matching positions exit
		// after a single comparison.
		if b[i] != '\r' {
			continue
		}
		if b[i+1] == '\n' && b[i+2] == '\r' && b[i+3] == '\n' {
			return i
		}
	}
	return -1
}

// equalFoldASCII reports whether a and b are equal, case-insensitive, ASCII only.
//
// HTTP header names are guaranteed ASCII per RFC 7230 §3.2 (token = 1*tchar,
// tchar excludes non-ASCII). This means we can fold case with a single
// arithmetic operation per byte and skip the unicode tables that
// strings.EqualFold drags in.
//
// Folding works by observing that in ASCII, 'A'..'Z' = 0x41..0x5A and
// 'a'..'z' = 0x61..0x7A — the only difference is bit 5 (0x20). For any
// letter, masking off bit 5 yields the uppercase form. For non-letters
// (digits, '-', etc.) the mask would corrupt the value, so we test the
// range first.
func equalFoldASCII(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca == cb {
			continue
		}
		// Fast inequality check above handles the common case where bytes
		// match exactly. Only fall through to the fold logic on mismatch.
		if 'A' <= ca && ca <= 'Z' {
			ca |= 0x20
		}
		if 'A' <= cb && cb <= 'Z' {
			cb |= 0x20
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// trimOWS returns b with leading and trailing optional whitespace removed.
// OWS is defined by RFC 7230 §3.2.3 as *( SP / HTAB ).
//
// Used when extracting a header field-value, which is formally:
//
//	header-field   = field-name ":" OWS field-value OWS
//
// We slice rather than copy — the returned []byte aliases the input.
func trimOWS(b []byte) []byte {
	start := 0
	for start < len(b) && (b[start] == ' ' || b[start] == '\t') {
		start++
	}
	end := len(b)
	for end > start && (b[end-1] == ' ' || b[end-1] == '\t') {
		end--
	}
	return b[start:end]
}

// parseUint parses a non-negative decimal integer from b. Returns the value
// and ok=true on success, or 0 and ok=false on any error (empty input,
// non-digit byte, overflow). No allocations, no error wrapping — callers
// translate ok=false into the appropriate HTTP-level error.
//
// Used for status codes (always 3 digits, fits in uint16) and, via the
// uint64 variant below, for Content-Length.
func parseUint(b []byte) (n uint32, ok bool) {
	if len(b) == 0 {
		return 0, false
	}
	for i := 0; i < len(b); i++ {
		c := b[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		// Overflow check. uint32 max is 4_294_967_295 (10 digits). We bail
		// if the next multiply-add would exceed that. This is exact, not
		// approximate — we only allow values that fit.
		if n > (^uint32(0)-uint32(c-'0'))/10 {
			return 0, false
		}
		n = n*10 + uint32(c-'0')
	}
	return n, true
}

// parseUint64 parses a non-negative decimal integer up to 64 bits.
// Used for Content-Length, which RFC 7230 §3.3.2 defines as an arbitrary
// non-negative integer. We cap at uint64 because larger bodies cannot
// physically be transferred over a 32-bit address space.
func parseUint64(b []byte) (n uint64, ok bool) {
	if len(b) == 0 {
		return 0, false
	}
	for i := 0; i < len(b); i++ {
		c := b[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		if n > (^uint64(0)-uint64(c-'0'))/10 {
			return 0, false
		}
		n = n*10 + uint64(c-'0')
	}
	return n, true
}

// parseHexUint parses a hexadecimal integer from b. Used for chunked
// transfer encoding chunk sizes, which are hex per RFC 7230 §4.1.
//
// The chunk-ext production after the size (";name=value") is rejected
// here — callers split on ';' before passing to this function.
//
// Hex digits are case-insensitive in the spec, so we accept both 'a'-'f'
// and 'A'-'F'.
func parseHexUint(b []byte) (n uint64, ok bool) {
	if len(b) == 0 {
		return 0, false
	}
	for i := 0; i < len(b); i++ {
		c := b[i]
		var d uint64
		switch {
		case '0' <= c && c <= '9':
			d = uint64(c - '0')
		case 'a' <= c && c <= 'f':
			d = uint64(c-'a') + 10
		case 'A' <= c && c <= 'F':
			d = uint64(c-'A') + 10
		default:
			return 0, false
		}
		// Overflow check: shifting by 4 must not lose the top nibble.
		if n > (^uint64(0))>>4 {
			return 0, false
		}
		n = n<<4 | d
	}
	return n, true
}

// writeUint writes the decimal representation of n into b, returning the
// number of bytes written. The caller must ensure b is large enough; for
// uint64 the worst case is 20 bytes ("18446744073709551615").
//
// Used by the request writer to emit Content-Length without going through
// strconv (which formats via a string return — extra allocation we avoid).
//
// Returns 0 and writes nothing if b is too small.
func writeUint(b []byte, n uint64) int {
	// Worst case 20 digits for uint64. Format right-to-left into a local
	// scratch, then copy left-to-right into the destination.
	var scratch [20]byte
	if n == 0 {
		if len(b) < 1 {
			return 0
		}
		b[0] = '0'
		return 1
	}
	i := len(scratch)
	for n > 0 {
		i--
		scratch[i] = byte('0' + n%10)
		n /= 10
	}
	written := len(scratch) - i
	if len(b) < written {
		return 0
	}
	copy(b, scratch[i:])
	return written
}

// bytesToString converts a []byte to a string without allocating. The
// returned string aliases the underlying byte storage — the caller MUST
// guarantee b is not mutated for the lifetime of the returned string.
//
// Used to expose header names and values from the response buffer as
// string-typed fields on Header without copying. The response buffer is
// caller-owned and the contract on Response is that the caller must keep
// it live (and not mutate it) until they're done with the Response, which
// makes this aliasing safe by construction.
//
// On stdlib Go (1.20+) this is unsafe.String(unsafe.SliceData, len). We
// use the same primitive here; sigo's runtime supports it via the
// standard unsafe package.
//
// Empty slices return "" without touching b — &b[0] would panic on a
// nil/empty slice.
func bytesToString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(unsafe.SliceData(b), len(b))
}
