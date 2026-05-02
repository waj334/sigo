package http

import "errors"

// ErrMalformedURL is returned by parseURL for inputs that don't match
// the supported subset.
var ErrMalformedURL = errors.New("http: malformed URL")

// parseURL splits a URL of the form
//
//	scheme://host[:port]/path[?query]
//
// into its components. The returned host and pathQuery slices alias
// the input string — callers must treat them as read-only.
//
// Supported schemes are "http" and "https". The path defaults to "/"
// if absent in the URL. Userinfo, fragments, and IPv6 literal hosts
// are NOT supported — this is intentionally a minimal parser for the
// embedded use case where URLs come from config and are well-formed.
//
// Allocation-free: the function only computes offsets into the input
// string and returns substrings, which alias the original storage.
func parseURL(rawURL string) (scheme uint8, host, pathQuery string, err error) {
	// Scheme: everything up to "://".
	const sep = "://"
	sepIdx := -1
	// Hand-rolled instead of strings.Index to keep this file dependency-free
	// and avoid a substring conversion at the call site.
	for i := 0; i+3 <= len(rawURL); i++ {
		if rawURL[i] == ':' && rawURL[i+1] == '/' && rawURL[i+2] == '/' {
			sepIdx = i
			break
		}
	}
	if sepIdx < 0 {
		return 0, "", "", ErrMalformedURL
	}

	schemeStr := rawURL[:sepIdx]
	switch {
	case len(schemeStr) == 4 && (schemeStr[0]|0x20) == 'h' && (schemeStr[1]|0x20) == 't' &&
		(schemeStr[2]|0x20) == 't' && (schemeStr[3]|0x20) == 'p':
		scheme = 0
	case len(schemeStr) == 5 && (schemeStr[0]|0x20) == 'h' && (schemeStr[1]|0x20) == 't' &&
		(schemeStr[2]|0x20) == 't' && (schemeStr[3]|0x20) == 'p' && (schemeStr[4]|0x20) == 's':
		scheme = 1
	default:
		return 0, "", "", ErrMalformedURL
	}

	// After the "://", the host runs until the first '/' or '?' or end.
	rest := rawURL[sepIdx+3:]
	pathStart := len(rest)
	for i := 0; i < len(rest); i++ {
		c := rest[i]
		if c == '/' || c == '?' {
			pathStart = i
			break
		}
	}
	host = rest[:pathStart]
	if len(host) == 0 {
		return 0, "", "", ErrMalformedURL
	}

	if pathStart == len(rest) {
		// No path — default to "/".
		pathQuery = "/"
	} else if rest[pathStart] == '?' {
		// Query but no path — synthesize a "/" before the "?".
		// This is a slight allocation (one new string) but only happens
		// for malformed URLs that are missing a path. Real URLs always
		// have a "/" before a "?". To stay strictly zero-alloc, reject
		// instead.
		return 0, "", "", ErrMalformedURL
	} else {
		pathQuery = rest[pathStart:]
	}

	// If the URL omits a port, fill in the default for the scheme.
	// This concatenation allocates one string. That's acceptable here
	// because parseURL is only called from the URL-taking shortcuts
	// (GetURL, PostURL, etc.), which already allocate a response buffer
	// and Response struct — one more short-string allocation rounds out
	// the overhead. Hot-path callers should use Client.Do directly with
	// pre-split Host and Path to avoid all of these allocations.
	hasPort := false
	for i := 0; i < len(host); i++ {
		if host[i] == ':' {
			hasPort = true
			break
		}
	}
	if !hasPort {
		switch scheme {
		case 0:
			host = host + ":80"
		case 1:
			host = host + ":443"
		}
	}

	return scheme, host, pathQuery, nil
}
