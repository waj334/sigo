package net

import (
	"errors"
	"nonstandard"
	"sync"
	"time"
	"unsafe"
)

// Resolver performs DNS lookups via lwIP's built-in resolver.
//
// The actual DNS protocol work (query formatting, retransmission, cache,
// multiple-server failover) is done inside lwIP. This type is a thin
// Go-side adapter that:
//   - Bridges the async lwIP callback into a synchronous Go API.
//   - Adds a deadline on top of lwIP's internal retry logic.
//   - Provides a literal-IP fast path so dotted-decimal addresses don't
//     touch the resolver at all.
//
// A zero Resolver is usable. Configure Timeout to override the default.
type Resolver struct {
	// Timeout is the maximum time a Lookup will wait for an answer.
	// Zero means defaultDNSTimeout (5s). lwIP's internal retry timing
	// is independent of this — Timeout is a hard upper bound from the
	// goroutine's perspective.
	Timeout time.Duration

	// lookupFn is the actual lookup implementation. Defaults to the
	// real lwIP-backed lookup. Tests swap it out.
	lookupFn func(name string, timeout time.Duration) ([4]byte, error)
}

const defaultDNSTimeout = 5 * time.Second

// DefaultResolver is the package-level Resolver used by LookupIPv4 and
// by the Dialer when given a hostname.
var DefaultResolver = &Resolver{}

// DNS-related errors. All exported as sentinels so callers can match
// with errors.Is.
var (
	ErrDNSNotConfigured = errors.New("net: no DNS server configured (DHCP not complete?)")
	ErrDNSTimeout       = errors.New("net: DNS lookup timed out")
	ErrDNSNotFound      = errors.New("net: host not found")
	ErrDNSFailed        = errors.New("net: DNS lookup failed")
)

// LookupIPv4 resolves name to a single IPv4 address.
//
// If name is already a dotted-decimal IPv4 literal, it's parsed and
// returned with no DNS traffic. Otherwise the lwIP resolver is invoked.
//
// Returns ErrDNSNotConfigured immediately if no DNS server is set
// (which on a DHCP-configured system means DHCP hasn't completed). The
// caller is expected to wait for network readiness before issuing
// hostname-based lookups.
func (r *Resolver) LookupIPv4(name string) ([4]byte, error) {
	// Literal fast path. parseIPv4 is in conn.go — already exists.
	if ip, err := parseIPv4(name); err == nil {
		return ip, nil
	}

	// Reject empty or trivially-bad names without involving lwIP. Saves
	// a queueOperation round-trip on garbage input.
	if len(name) == 0 || len(name) > 253 {
		return [4]byte{}, ErrDNSFailed
	}

	timeout := r.Timeout
	if timeout == 0 {
		timeout = defaultDNSTimeout
	}

	fn := r.lookupFn
	if fn == nil {
		fn = lwipLookupIPv4
	}
	return fn(name, timeout)
}

// LookupIPv4 is the package-level shortcut using DefaultResolver.
func LookupIPv4(name string) ([4]byte, error) {
	return DefaultResolver.LookupIPv4(name)
}

// dnsRequest holds the state for one in-flight DNS query. Instances
// live in a fixed slot pool (dnsSlots below) — they're never GC'd
// during the program lifetime, which avoids the "lwIP holds our
// pointer past Go-side timeout" hazard.
type dnsRequest struct {
	// Set by the issuing goroutine before kicking off the lookup.
	// Read by the lwIP callback (under the slot mutex).
	name  string
	nameC []byte // null-terminated copy of name; lifetime tied to slot

	// Set by the lwIP callback. The issuing goroutine reads these
	// after the done channel fires (or the timeout expires).
	addr   [4]byte
	failed bool

	// done is signaled by the callback (or by the issuer's timeout
	// path, to prevent the slot from being freed while lwIP still has
	// our pointer). Buffered with capacity 1 so the callback never
	// blocks even if the issuer has already given up and moved on.
	done chan struct{}

	// abandoned is set by the issuer when it gives up waiting (timeout).
	// The callback checks this before signaling — if abandoned, the
	// callback releases the slot itself rather than handing it back to
	// the issuer.
	abandoned bool
}

// dnsSlot wraps a dnsRequest with a mutex and an in-use bit. The pool
// is small because real concurrent DNS demand on embedded is small —
// even a chatty IoT device rarely has more than 1-2 lookups in flight.
type dnsSlot struct {
	mu      sync.Mutex
	inUse   bool
	request dnsRequest
}

// Eight slots is enough for any realistic embedded workload. Each slot
// is small (a few hundred bytes) so the total fixed cost is negligible
// next to lwIP's own DNS table.
const maxDNSSlots = 8

var dnsSlots [maxDNSSlots]dnsSlot

// acquireDNSSlot returns an unused slot index, or -1 if the pool is full.
func acquireDNSSlot() int {
	for i := 0; i < maxDNSSlots; i++ {
		dnsSlots[i].mu.Lock()
		if !dnsSlots[i].inUse {
			dnsSlots[i].inUse = true
			// Reset previous state. We don't shrink the nameC buffer —
			// reusing it across lookups is the whole point of the pool.
			dnsSlots[i].request.failed = false
			dnsSlots[i].request.abandoned = false
			dnsSlots[i].request.addr = [4]byte{}
			// Lazily allocate the done channel on first use.
			if dnsSlots[i].request.done == nil {
				dnsSlots[i].request.done = make(chan struct{}, 1)
			} else {
				// Drain any stale signal from a previous late callback.
				select {
				case <-dnsSlots[i].request.done:
				default:
				}
			}
			dnsSlots[i].mu.Unlock()
			return i
		}
		dnsSlots[i].mu.Unlock()
	}
	return -1
}

// releaseDNSSlot marks a slot as available for reuse.
func releaseDNSSlot(idx int) {
	dnsSlots[idx].mu.Lock()
	dnsSlots[idx].inUse = false
	dnsSlots[idx].mu.Unlock()
}

// lwipLookupIPv4 is the real DNS lookup backed by lwIP. Swapped out in
// tests via Resolver.lookupFn.
func lwipLookupIPv4(name string, timeout time.Duration) ([4]byte, error) {
	// Check that DNS is actually configured. Without this we'd issue a
	// lookup that fails synchronously inside lwIP, which is correct but
	// the error isn't very informative ("Illegal value"). Surface a
	// clear error instead.
	if dnsGetServerV4(0) == ([4]byte{}) {
		return [4]byte{}, ErrDNSNotConfigured
	}

	idx := acquireDNSSlot()
	if idx < 0 {
		// Pool exhausted. Caller can retry.
		return [4]byte{}, errors.New("net: DNS slot pool full")
	}

	slot := &dnsSlots[idx]
	req := &slot.request

	// Pin the name as a null-terminated C string in the slot's buffer.
	// The buffer persists across lookups, so we just resize it in place.
	needed := len(name) + 1
	if cap(req.nameC) < needed {
		req.nameC = make([]byte, needed)
	} else {
		req.nameC = req.nameC[:needed]
	}
	copy(req.nameC, name)
	req.nameC[len(name)] = 0
	req.name = name

	// DIAG: dump DNS server and queried name before issuing the lookup.
	{
		s := dnsGetServerV4(0)
		print("DNS server: ")
		print(s[0])
		print(".")
		print(s[1])
		print(".")
		print(s[2])
		print(".")
		print(s[3])
		print("\n")
		print("DNS query name bytes (")
		print(len(req.nameC))
		print("):")
		for _, b := range req.nameC {
			print(" ")
			print(b)
		}
		print("\n")
	}

	var addr ipAddr

	// Issue the lookup on the lwIP goroutine. The callback may fire
	// synchronously (cache hit) inside this queueOperation, in which
	// case dnsGetHostByName returns errOk and addr is already filled.
	// Otherwise it returns errInProgress and the callback fires later.
	var issueErr lwipError
	queueOperation(func() {
		issueErr = dnsGetHostByName(
			unsafe.Pointer(&req.nameC[0]),
			&addr,
			nonstandard.PointerOf(dnsFound),
			unsafe.Pointer(req),
		)
		if issueErr == errOk {
			// Cache hit — addr is valid right now. Read it on the lwIP
			// goroutine while we're still here, since once we return,
			// addr is on this closure's stack.
			req.addr = ipAddrGetV4(&addr)
			// Fall through; we'll signal done outside the closure.
		}
	})

	switch issueErr {
	case errOk:
		// Cache hit; addr was extracted on the lwIP goroutine.
		ip := req.addr
		releaseDNSSlot(idx)
		return ip, nil
	case errInProgress:
		// Async path; wait for the callback.
	default:
		releaseDNSSlot(idx)
		// errValue from lwIP usually means "no DNS server" but we
		// pre-checked that. Treat any other error as a generic failure.
		return [4]byte{}, ErrDNSFailed
	}

	// Wait for the callback or timeout.
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-req.done:
		// Callback fired. Read result, release slot, return.
		var ip [4]byte
		failed := false
		slot.mu.Lock()
		ip = req.addr
		failed = req.failed
		slot.mu.Unlock()
		releaseDNSSlot(idx)
		if failed {
			return [4]byte{}, ErrDNSNotFound
		}
		return ip, nil

	case <-timer.C:
		// Timed out — but a callback might have fired between the timer
		// firing and us reaching this branch. Take the lock and check
		// the done channel: if the callback won, consume its result;
		// otherwise mark abandoned so the (eventual) late callback
		// releases the slot rather than the issuer.
		slot.mu.Lock()
		select {
		case <-req.done:
			// Callback won the race. Use its result.
			ip := req.addr
			failed := req.failed
			slot.mu.Unlock()
			releaseDNSSlot(idx)
			if failed {
				return [4]byte{}, ErrDNSNotFound
			}
			return ip, nil
		default:
			// Genuinely timed out. Mark abandoned and leave the slot
			// in-use. The late callback (if it ever fires) will see
			// abandoned=true and release the slot itself.
			req.abandoned = true
			slot.mu.Unlock()
			return [4]byte{}, ErrDNSTimeout
		}
	}
}

// dnsFound is the C-side dns_found_callback. Exported so lwIP can call it.
//
// We don't use the `name` argument — the dnsRequest struct already knows
// its own name — so it's typed as unsafe.Pointer to dodge the *C.char
// type and keep this file independent of lwip_dns.go's CGo machinery.
//
//go:export dnsFound dns_found_callback
func dnsFound(name unsafe.Pointer, ipaddr *ipAddr, arg unsafe.Pointer) {
	_ = name
	// DIAG: dump what the callback actually receives.
	{
		ip := ipAddrGetV4(ipaddr)
		print("dnsFound: name=")
		print(uintptr(name))
		print(" ipaddr=")
		print(uintptr(unsafe.Pointer(ipaddr)))
		print(" arg=")
		print(uintptr(arg))
		print(" parsed=")
		print(ip[0])
		print(".")
		print(ip[1])
		print(".")
		print(ip[2])
		print(".")
		print(ip[3])
		print("\n")
	}
	req := (*dnsRequest)(arg)

	// Find the slot containing this request. We could store the slot
	// index in arg directly, but using the request pointer keeps the
	// dnsGetHostByName call site simpler. Eight slots — linear scan
	// is fine.
	var slot *dnsSlot
	for i := 0; i < maxDNSSlots; i++ {
		if &dnsSlots[i].request == req {
			slot = &dnsSlots[i]
			break
		}
	}
	if slot == nil {
		// Stray callback for a request that's not in our pool. Ignore.
		return
	}

	slot.mu.Lock()
	if req.abandoned {
		// The issuer gave up. We're responsible for releasing the slot.
		slot.mu.Unlock()
		releaseDNSSlot(slotIndexOf(slot))
		return
	}

	if ipaddr == nil {
		req.failed = true
	} else {
		req.addr = ipAddrGetV4(ipaddr)
	}
	slot.mu.Unlock()

	// Signal the issuer. Buffered channel; never blocks.
	select {
	case req.done <- struct{}{}:
	default:
	}
}

func slotIndexOf(s *dnsSlot) int {
	for i := 0; i < maxDNSSlots; i++ {
		if &dnsSlots[i] == s {
			return i
		}
	}
	return -1
}
