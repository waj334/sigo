package http

import (
	"sync"
	"time"

	"net"
)

// maxPoolSlots is the fixed capacity of the idle connection pool.
//
// Sized for typical embedded workloads: one or two HTTP endpoints, one
// request in flight at a time. Increasing this trades SRAM (each slot
// holds a TCPConn with three channels, ~200 bytes plus lwIP pcb state)
// for fewer handshakes when talking to many distinct hosts.
const maxPoolSlots = 4

// defaultIdleTTL is how long a released conn may sit in the pool before
// it's considered stale and closed on next access.
//
// Chosen to expire before typical HTTP server keep-alive timeouts (60-75s
// is common). If we expire first, we avoid the race where the server
// closes the conn between our check and our write — that race produces
// the "first write fails on pooled conn" symptom that retry-on-stale
// exists to paper over. Shorter TTL means fewer such races at the cost
// of more handshakes.
const defaultIdleTTL = 30 * time.Second

// connKey identifies which pool slots are interchangeable.
//
// Two requests can share a conn iff they target the same host:port AND
// use the same scheme. We don't share TLS conns across hosts even if
// the underlying TCP conn could be reused — the TLS session is bound
// to a specific peer.
type connKey struct {
	host   string // exactly the string passed to net.Dial; e.g. "192.168.1.10:8080"
	scheme uint8  // 0 = http, 1 = https
}

// pooledConn is the per-slot bookkeeping. When inUse[i] is false, this
// slot's conn is idle and may be returned by get(). When inUse[i] is
// true, the conn has been handed out and must be returned via release()
// or discard().
//
// An empty slot has conn == nil; it is neither in-use nor idle.
type pooledConn struct {
	conn       net.Conn
	key        connKey
	releasedMs uint32 // millisecond tick when this conn was last released to the pool
}

// connPool is a fixed-capacity LRU cache of idle HTTP connections.
//
// Operations are O(N) over the slot array. With N=4 this is faster than
// a map both in cycles and in allocations (no rehashing, no hash
// computation). The slots array lives inline in the pool struct, so the
// whole thing is a single allocation owned by the Client.
//
// Concurrency: a single mutex guards all fields. Operations execute
// quickly (microseconds) and are not on a latency-critical path —
// contention only matters if multiple goroutines are issuing requests
// concurrently, which is unusual on embedded.
type connPool struct {
	mu      sync.Mutex
	slots   [maxPoolSlots]pooledConn
	inUse   [maxPoolSlots]bool
	idleTTL time.Duration // 0 means no TTL check (pool entries never expire)
}

// get returns an idle conn matching key, or (nil, -1) if none is available.
//
// On hit, the returned slot index must eventually be passed to release()
// or discard() — it identifies which slot owns the conn so we don't have
// to search by pointer on return.
//
// On miss, the caller must dial a fresh conn and pass slot=-1 to
// release() when done; release() will assign a slot at that point.
//
// Stale conns (idle longer than idleTTL) are closed and the slot freed
// before we return — get() will not hand back an expired conn. Whether
// the underlying socket is *actually* still alive we cannot know without
// reading from it, which is why the codec also retries on write failure
// for idempotent methods.
func (p *connPool) get(key connKey, nowMs uint32) (net.Conn, int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	ttlMs := uint32(p.idleTTL / time.Millisecond)

	for i := 0; i < maxPoolSlots; i++ {
		if p.inUse[i] || p.slots[i].conn == nil {
			continue
		}
		// Expire stale entries lazily. The subtraction is wrap-safe for
		// the 49-day cycle of uint32 millis as long as idleTTL is well
		// under that bound, which it always is.
		if ttlMs != 0 && nowMs-p.slots[i].releasedMs > ttlMs {
			p.closeSlotLocked(i)
			continue
		}
		if p.slots[i].key == key {
			p.inUse[i] = true
			return p.slots[i].conn, i
		}
	}
	return nil, -1
}

// release returns a conn to the pool.
//
// If slot >= 0, the conn was previously obtained from get() at that slot;
// release marks the slot idle again and updates its timestamp.
//
// If slot == -1, the conn was freshly dialed. release finds a free slot,
// or if all slots are occupied evicts the LRU idle entry, and stores
// this conn there. The eviction may close another conn (which goes
// through queueOperation in your TCPConn implementation), so this call
// can briefly block.
//
// After calling release, the caller MUST NOT touch conn — its ownership
// has transferred to the pool, which may close it during a future
// eviction or expiry.
func (p *connPool) release(conn net.Conn, key connKey, slot int, nowMs uint32) {
	p.mu.Lock()

	if slot >= 0 && slot < maxPoolSlots && p.slots[slot].conn == conn {
		// Returning a conn we previously handed out. Mark idle and stamp.
		p.inUse[slot] = false
		p.slots[slot].releasedMs = nowMs
		p.mu.Unlock()
		return
	}

	// Fresh conn or stale slot reference — find a home for it.

	// First pass: empty slot.
	for i := 0; i < maxPoolSlots; i++ {
		if p.slots[i].conn == nil {
			p.slots[i] = pooledConn{
				conn:       conn,
				key:        key,
				releasedMs: nowMs,
			}
			p.inUse[i] = false
			p.mu.Unlock()
			return
		}
	}

	// Second pass: evict the LRU idle slot.
	//
	// Find the idle slot with the oldest releasedMs. If every slot is
	// in-use, we have nowhere to put this conn — close it and move on.
	// This shouldn't happen in practice (we only release after a
	// request completes, and requests don't outnumber slots) but the
	// pool must remain correct under contention.
	oldest := -1
	var oldestAge uint32
	for i := 0; i < maxPoolSlots; i++ {
		if p.inUse[i] {
			continue
		}
		age := nowMs - p.slots[i].releasedMs
		if oldest == -1 || age > oldestAge {
			oldest = i
			oldestAge = age
		}
	}

	if oldest == -1 {
		// All slots in-use. Drop conn on the floor (close without pooling).
		p.mu.Unlock()
		_ = conn.Close()
		return
	}

	// Close the evicted conn before installing the new one. We hold the
	// mutex across Close, which is acceptable because Close is fast and
	// the alternative — releasing the lock to Close, then reacquiring —
	// opens a window where another goroutine could grab the slot.
	evicted := p.slots[oldest].conn
	p.slots[oldest] = pooledConn{
		conn:       conn,
		key:        key,
		releasedMs: nowMs,
	}
	p.inUse[oldest] = false
	p.mu.Unlock()

	// Close outside the mutex: TCPConn.Close calls queueOperation, which
	// can park the goroutine waiting for the lwIP scheduler. Holding the
	// pool mutex across that wait would serialize all pool operations
	// behind the lwIP goroutine, which is exactly what we don't want.
	_ = evicted.Close()
}

// discard removes a conn from the pool and closes it.
//
// Use this when the conn is known to be broken — write failed, peer
// closed, protocol error, etc. After discard returns, the slot (if any)
// is empty and available for the next release().
//
// Passing slot == -1 closes the conn without touching the pool, which
// is correct for a freshly dialed conn that errored before ever being
// pooled.
func (p *connPool) discard(conn net.Conn, slot int) {
	if slot >= 0 && slot < maxPoolSlots {
		p.mu.Lock()
		if p.slots[slot].conn == conn {
			p.slots[slot] = pooledConn{}
			p.inUse[slot] = false
		}
		p.mu.Unlock()
	}
	_ = conn.Close()
}

// closeAll closes every conn in the pool and empties all slots.
//
// Called from Client.CloseIdleConnections. Useful before a deep-sleep
// transition, or in tests, or when reconfiguring the network stack.
// Conns currently in-use (handed out via get) are NOT closed — their
// owners are responsible for them and will return them via release or
// discard as usual.
func (p *connPool) closeAll() {
	p.mu.Lock()
	// Snapshot the conns to close so we can release the mutex before
	// calling Close (same reasoning as in release: Close goes through
	// queueOperation).
	var toClose [maxPoolSlots]net.Conn
	n := 0
	for i := 0; i < maxPoolSlots; i++ {
		if p.slots[i].conn != nil && !p.inUse[i] {
			toClose[n] = p.slots[i].conn
			n++
			p.slots[i] = pooledConn{}
		}
	}
	p.mu.Unlock()

	for i := 0; i < n; i++ {
		_ = toClose[i].Close()
	}
}

// closeSlotLocked closes the conn in slot i and empties the slot.
// Caller must hold p.mu. The Close() call happens with the lock held —
// see release() for the rationale on when this is acceptable. For lazy
// expiry inside get(), we accept the brief stall because the alternative
// (release lock, close, reacquire, re-validate state) is significantly
// more complex and the stall is measured in tens of microseconds.
func (p *connPool) closeSlotLocked(i int) {
	conn := p.slots[i].conn
	p.slots[i] = pooledConn{}
	p.inUse[i] = false
	if conn != nil {
		_ = conn.Close()
	}
}

// nowMs returns the current millisecond tick used for idle-TTL bookkeeping.
//
// uint32 wraps every ~49 days. Subtraction in the wrapped domain is
// correct as long as the difference being measured is much smaller than
// the wrap period, which idle TTLs trivially satisfy.
func nowMs() uint32 {
	return uint32(time.Now().UnixMilli())
}
