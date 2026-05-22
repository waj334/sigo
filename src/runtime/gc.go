package runtime

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	//sigo:extern __gc_scan_start __gc_scan_start
	__gc_scan_start unsafe.Pointer

	//sigo:extern __gc_scan_end __gc_scan_end
	__gc_scan_end unsafe.Pointer

	//sigo:extern __heap_start __heap_start
	__heap_start unsafe.Pointer

	//sigo:extern __heap_end __heap_end
	__heap_end unsafe.Pointer

	//sigo:extern __stack_top __stack_top
	__stack_top unsafe.Pointer

	//sigo:extern __stack_bottom __stack_bottom
	__stack_bottom unsafe.Pointer

	gc   _gc
	gcMu sync.Mutex // Protects GC data structures

	// Deferred barrier queue for when we can't acquire the lock (e.g., from ISR)
	barrierQueue     [64]uintptr // Circular buffer of pointers needing shading
	barrierQueueHead uint32
	barrierQueueTail uint32
)

const (
	gcWordSize     = unsafe.Sizeof(uintptr(0))
	gcPointerAlign = unsafe.Alignof(uintptr(0))
	gcObjectSize   = unsafe.Sizeof(gcObject{})

	// Bounded work per incremental step.
	// These values balance GC throughput vs. interrupt latency.
	// Lower values = better interrupt response, but slower GC.
	gcMarkBatch  = 64 // words to scan per root scan step
	gcGrayBatch  = 16 // gray objects to process per step
	gcSweepBatch = 8  // objects to sweep per step (reduced for lower latency)
)

// --------------------------------------------------------------------------
// Object header
// --------------------------------------------------------------------------

type gcColor uint8

const (
	gcWhite gcColor = iota
	gcGray
	gcBlack
)

type gcObject struct {
	next     *gcObject // main object list
	grayNext *gcObject // gray worklist linkage
	size     uintptr   // user data size (excluding this header)
	color    gcColor
}

// --------------------------------------------------------------------------
// GC phases and sub-states
// --------------------------------------------------------------------------

type gcPhase uint8

const (
	gcIdle gcPhase = iota
	gcMark
	gcRemark
	gcSweep
)

type gcScanState uint8

const (
	gcScanStack gcScanState = iota
	gcScanGoroutines
	gcScanGlobals
	gcScanGray
)

// --------------------------------------------------------------------------
// Bitmap -- one bit per pointer-aligned heap slot
// --------------------------------------------------------------------------

type gcBitmap struct {
	words    unsafe.Pointer
	numWords uintptr
}

func (bm *gcBitmap) init(numSlots uintptr) {
	bm.numWords = (numSlots + 31) / 32
	byteSize := bm.numWords * unsafe.Sizeof(uint32(0))
	bm.words = malloc(byteSize)
	if bm.words == nil {
		abort()
	}
	for i := uintptr(0); i < bm.numWords; i++ {
		bm.wordAt(i).set(0)
	}
}

//go:inline
func (bm *gcBitmap) wordAt(index uintptr) *gcBitmapWord {
	return (*gcBitmapWord)(unsafe.Add(bm.words, index*unsafe.Sizeof(uint32(0))))
}

//go:inline
func (bm *gcBitmap) set(slot uintptr) { bm.wordAt(slot / 32).setBit(slot % 32) }

//go:inline
func (bm *gcBitmap) clear(slot uintptr) { bm.wordAt(slot / 32).clearBit(slot % 32) }

//go:inline
func (bm *gcBitmap) test(slot uintptr) bool { return bm.wordAt(slot / 32).testBit(slot % 32) }

func (bm *gcBitmap) setRange(start, count uintptr) {
	for i := uintptr(0); i < count; i++ {
		bm.set(start + i)
	}
}

func (bm *gcBitmap) clearRange(start, count uintptr) {
	for i := uintptr(0); i < count; i++ {
		bm.clear(start + i)
	}
}

func (bm *gcBitmap) findPrevSet(slot uintptr) (uintptr, bool) {
	wordIdx := slot / 32
	bitIdx := slot % 32
	w := bm.wordAt(wordIdx).get()
	masked := w & ((1 << (bitIdx + 1)) - 1)
	if masked != 0 {
		return wordIdx*32 + gcHighBit(masked), true
	}
	for wordIdx > 0 {
		wordIdx--
		w = bm.wordAt(wordIdx).get()
		if w != 0 {
			return wordIdx*32 + gcHighBit(w), true
		}
	}
	return 0, false
}

type gcBitmapWord uint32

//go:inline
func (w *gcBitmapWord) get() uint32 { return uint32(*w) }

//go:inline
func (w *gcBitmapWord) set(v uint32) { *w = gcBitmapWord(v) }

//go:inline
func (w *gcBitmapWord) setBit(bit uintptr) { *w |= gcBitmapWord(1 << bit) }

//go:inline
func (w *gcBitmapWord) clearBit(bit uintptr) { *w &^= gcBitmapWord(1 << bit) }

//go:inline
func (w *gcBitmapWord) testBit(bit uintptr) bool { return (uint32(*w) & (1 << bit)) != 0 }

func gcHighBit(v uint32) uintptr {
	n := uintptr(0)
	if v&0xFFFF0000 != 0 {
		n += 16
		v >>= 16
	}
	if v&0xFF00 != 0 {
		n += 8
		v >>= 8
	}
	if v&0xF0 != 0 {
		n += 4
		v >>= 4
	}
	if v&0xC != 0 {
		n += 2
		v >>= 2
	}
	if v&0x2 != 0 {
		n += 1
	}
	return n
}

// --------------------------------------------------------------------------
// GC state
// --------------------------------------------------------------------------

type _gc struct {
	head *gcObject // main object list

	// Gray worklist -- singly linked via grayNext.
	grayList *gcObject

	// Incremental root scanning state.
	scanState        gcScanState
	currentAddress   uintptr
	endAddress       uintptr
	currentGoroutine *goroutine

	// Incremental sweep state.
	sweepCurr *gcObject
	sweepPrev *gcObject

	// Bitmaps for O(1) object lookup.
	allocBitmap gcBitmap
	startBitmap gcBitmap
	heapBase    uintptr
	numSlots    uintptr

	phase gcPhase
}

// --------------------------------------------------------------------------
// Bitmap registration
// --------------------------------------------------------------------------

//go:inline
func (gc *_gc) slotIndex(addr uintptr) uintptr {
	return (addr - gc.heapBase) / gcPointerAlign
}

//go:nosplit
//go:nowritebarrier
func (gc *_gc) registerObject(obj *gcObject) {
	dataStart := uintptr(unsafe.Pointer(obj)) + gcObjectSize
	startSlot := gc.slotIndex(dataStart)
	slotCount := obj.size / gcPointerAlign
	gc.startBitmap.set(startSlot)
	gc.allocBitmap.setRange(startSlot, slotCount)
}

//go:nosplit
//go:nowritebarrier
func (gc *_gc) deregisterObject(obj *gcObject) {
	dataStart := uintptr(unsafe.Pointer(obj)) + gcObjectSize
	startSlot := gc.slotIndex(dataStart)
	slotCount := obj.size / gcPointerAlign
	gc.startBitmap.clear(startSlot)
	gc.allocBitmap.clearRange(startSlot, slotCount)
}

// --------------------------------------------------------------------------
// O(1) object lookup
// --------------------------------------------------------------------------

//go:nosplit
//go:nowritebarrier
func (gc *_gc) findObject(val uintptr) *gcObject {
	if val < gc.heapBase || val >= gc.heapBase+gc.numSlots*gcPointerAlign {
		return nil
	}

	slot := gc.slotIndex(val)
	if !gc.allocBitmap.test(slot) {
		return nil
	}

	startSlot, found := gc.startBitmap.findPrevSet(slot)
	if !found {
		return nil
	}

	dataStart := gc.heapBase + startSlot*gcPointerAlign

	// Guard: object header must be inside the heap.
	if dataStart < gc.heapBase+gcObjectSize {
		return nil
	}

	obj := (*gcObject)(unsafe.Pointer(dataStart - gcObjectSize))
	if val >= dataStart && val < dataStart+obj.size {
		return obj
	}
	return nil
}

// --------------------------------------------------------------------------
// Write barrier
// --------------------------------------------------------------------------
//
// This GC uses an incremental tricolor marking algorithm with a Yuasa-style
// insertion barrier. The barrier ensures that:
//
// 1. Any pointer stored into a heap object during marking is shaded (marked gray)
// 2. New allocations during marking are marked black (insertion barrier)
// 3. This maintains the tricolor invariant: no black object points to a white object
//
// The barrier is only active during the Mark and Remark phases. During Idle and
// Sweep phases, stores proceed without barriers for better performance.
//
//go:inline
func gcBarrierActive() bool {
	return gc.phase == gcMark || gc.phase == gcRemark
}

// PRECONDITION: gcMu is held.
//
//go:nosplit
//go:nowritebarrier
func (gc *_gc) shade(obj *gcObject) {
	if obj.color == gcWhite {
		obj.color = gcGray
		obj.grayNext = gc.grayList
		gc.grayList = obj
	}
}

//go:nosplit
//go:nowritebarrier
func (gc *_gc) shadeSafe(obj *gcObject) {
	gcMu.Lock()
	gc.shade(obj)
	gcMu.Unlock()
}

// The compiler must lower stores into GC-managed heap memory through this helper.
//
// slot must point at a machine-word slot in heap memory.
// val is the word being written.
//
//go:export gcWriteBarrier runtime.gcWriteBarrier
//go:nosplit
//go:nowritebarrier
func gcWriteBarrier(slot *uintptr, val uintptr) {
	// Fast path: if GC is idle, just do the store
	if !gcBarrierActive() {
		*slot = val
		return
	}

	// Store the value first
	*slot = val

	// Shade the new value to prevent it from being collected prematurely
	// This implements an insertion barrier (Yuasa-style)
	if val != 0 { // Only check non-nil pointers
		// Try to shade the object. If we're in an interrupt or can't get the lock,
		// queue it for later processing.
		if !gcTryShade(val) {
			// Failed to shade immediately - queue for deferred processing
			gcQueueBarrier(val)
		}
	}
}

// gcTryShade attempts to shade an object without blocking.
// Returns true if successful, false if the object needs to be queued.
//
//go:nosplit
//go:nowritebarrier
func gcTryShade(val uintptr) bool {
	// Try to acquire the lock without blocking
	if !gcMu.TryLock() {
		return false
	}

	// Find and shade the object
	child := gc.findObject(val)
	if child != nil {
		gc.shade(child)
	}

	gcMu.Unlock()
	return true
}

// gcQueueBarrier queues a pointer for deferred barrier processing.
// This is used when we can't immediately acquire the GC lock (e.g., from ISR).
//
//go:nosplit
//go:nowritebarrier
func gcQueueBarrier(val uintptr) {
	// Lock-free circular buffer insertion
	for {
		head := atomic.LoadUint32(&barrierQueueHead)
		tail := atomic.LoadUint32(&barrierQueueTail)

		// Check if queue is full
		nextTail := (tail + 1) % uint32(len(barrierQueue))
		if nextTail == head {
			// Queue full - this is a critical error, but we can't block
			// The object might be collected, but this is better than deadlock
			// In practice, the queue should be large enough and flushed frequently
			return
		}

		// Try to claim this slot
		if atomic.CompareAndSwapUint32(&barrierQueueTail, tail, nextTail) {
			barrierQueue[tail] = val
			return
		}
		// CAS failed, retry
	}
}

// gcFlushBarrierQueue processes any queued barrier entries.
// PRECONDITION: gcMu is held.
//
//go:nosplit
//go:nowritebarrier
func gcFlushBarrierQueue() {
	head := atomic.LoadUint32(&barrierQueueHead)
	tail := atomic.LoadUint32(&barrierQueueTail)

	for head != tail {
		val := barrierQueue[head]
		if child := gc.findObject(val); child != nil {
			gc.shade(child)
		}

		nextHead := (head + 1) % uint32(len(barrierQueue))
		atomic.StoreUint32(&barrierQueueHead, nextHead)
		head = nextHead
	}
}

//go:inline
func gcWriteBarrierPtr(slot *unsafe.Pointer, val unsafe.Pointer) {
	gcWriteBarrier((*uintptr)(unsafe.Pointer(slot)), uintptr(val))
}

// Barriered bulk copy into heap memory. The compiler should use this for bulk
// moves/copies whose destination is in the GC heap.
//
//go:export gcWriteBarrierCopy runtime.gcWriteBarrierCopy
//go:nosplit
//go:nowritebarrier
func gcWriteBarrierCopy(dst, src unsafe.Pointer, n uintptr) {
	if n == 0 {
		return
	}

	// Fast path: if GC is idle, just do the copy
	if !gcBarrierActive() {
		for i := uintptr(0); i < n; i++ {
			*(*byte)(unsafe.Add(dst, i)) = *(*byte)(unsafe.Add(src, i))
		}
		return
	}

	// Slow path: barrier each word-sized slot
	// Process word-aligned data through the write barrier
	wordCount := n / gcWordSize
	for i := uintptr(0); i < wordCount; i++ {
		val := *(*uintptr)(unsafe.Add(src, i*gcWordSize))
		gcWriteBarrier((*uintptr)(unsafe.Add(dst, i*gcWordSize)), val)
	}

	// Copy any remaining bytes (tail that's not word-aligned)
	// These can't be pointers since pointers must be word-aligned
	// No lock needed - these are non-pointer bytes
	for i := wordCount * gcWordSize; i < n; i++ {
		*(*byte)(unsafe.Add(dst, i)) = *(*byte)(unsafe.Add(src, i))
	}
}

// --------------------------------------------------------------------------
// Scanning primitives
// --------------------------------------------------------------------------

// scanObject scans an object's data for heap pointers.
// This form is used during incremental scanning with interrupts enabled.
func (gc *_gc) scanObject(obj *gcObject) {
	dataStart := uintptr(unsafe.Pointer(obj)) + gcObjectSize
	for addr := dataStart; addr < dataStart+obj.size; addr += gcWordSize {
		val := *(*uintptr)(unsafe.Pointer(addr))
		if child := gc.findObject(val); child != nil {
			gc.shadeSafe(child)
		}
	}
}

// scanObjectAtomic scans an object while interrupts are already disabled.
func (gc *_gc) scanObjectAtomic(obj *gcObject) {
	dataStart := uintptr(unsafe.Pointer(obj)) + gcObjectSize
	for addr := dataStart; addr < dataStart+obj.size; addr += gcWordSize {
		val := *(*uintptr)(unsafe.Pointer(addr))
		if child := gc.findObject(val); child != nil {
			gc.shade(child)
		}
	}
}

// --------------------------------------------------------------------------
// Incremental root scanning
// --------------------------------------------------------------------------

func (gc *_gc) scanRootsIncremental() bool {
	for i := 0; i < gcMarkBatch && gc.currentAddress < gc.endAddress; i++ {
		val := *(*uintptr)(unsafe.Pointer(gc.currentAddress))
		if obj := gc.findObject(val); obj != nil {
			gc.shadeSafe(obj)
		}
		gc.currentAddress += gcWordSize
	}

	if gc.currentAddress >= gc.endAddress {
		return gc.advanceScanState()
	}
	return false
}

func (gc *_gc) advanceScanState() bool {
	switch gc.scanState {
	case gcScanStack:
		gc.scanState = gcScanGoroutines
		if headGoroutine != nil {
			gc.currentGoroutine = headGoroutine
			gc.currentAddress, gc.endAddress = gcGoroutineStack(gc.currentGoroutine)
		} else {
			return gc.advanceScanState()
		}

	case gcScanGoroutines:
		if gc.currentGoroutine.next != headGoroutine {
			gc.currentGoroutine = gc.currentGoroutine.next
			gc.currentAddress, gc.endAddress = gcGoroutineStack(gc.currentGoroutine)
		} else {
			gc.scanState = gcScanGlobals
			gc.currentAddress = gcGlobalsStart()
			gc.endAddress = gcGlobalsEnd()
		}

	case gcScanGlobals:
		gc.scanState = gcScanGray
		return true
	}
	return false
}

// --------------------------------------------------------------------------
// Incremental gray processing
// --------------------------------------------------------------------------

func (gc *_gc) processGrayIncremental() bool {
	for i := 0; i < gcGrayBatch; i++ {
		gcMu.Lock()

		// Flush any queued barriers first
		gcFlushBarrierQueue()

		obj := gc.grayList
		if obj != nil {
			gc.grayList = obj.grayNext
			obj.grayNext = nil
			obj.color = gcBlack
		}

		gcMu.Unlock()

		if obj == nil {
			return true
		}

		gc.scanObject(obj)
	}

	gcMu.Lock()
	gcFlushBarrierQueue() // Flush again before checking if we're done
	isEmpty := gc.grayList == nil
	gcMu.Unlock()
	return isEmpty
}

// --------------------------------------------------------------------------
// Mark phase -- incremental
// --------------------------------------------------------------------------

func (gc *_gc) markStep() {
	switch gc.scanState {
	case gcScanStack, gcScanGoroutines, gcScanGlobals:
		gc.scanRootsIncremental()

	case gcScanGray:
		if gc.processGrayIncremental() {
			gc.phase = gcRemark
		}
	}
}

// --------------------------------------------------------------------------
// Remark phase -- stop-the-world final re-scan
// --------------------------------------------------------------------------

func (gc *_gc) remark() {
	// Remark phase is stop-the-world to ensure we see a consistent snapshot
	// Disable interrupts only during the critical sections to allow serial I/O
	// between scanning operations

	gcMu.Lock()

	// Flush any pending barriers before starting remark
	gcFlushBarrierQueue()

	state := DisableInterrupts()
	gc.scanRangeAtomic(gcStackBottom(), gcStackTop())
	EnableInterrupts(state)

	if headGoroutine != nil {
		g := headGoroutine
		for {
			state := DisableInterrupts()
			low, high := gcGoroutineStack(g)
			gc.scanRangeAtomic(low, high)
			EnableInterrupts(state)
			g = g.next
			if g == headGoroutine {
				break
			}
		}
	}

	state = DisableInterrupts()
	gc.scanRangeAtomic(gcGlobalsStart(), gcGlobalsEnd())
	EnableInterrupts(state)

	// Process remaining gray objects
	for {
		// Flush any barriers that arrived during processing
		gcFlushBarrierQueue()

		if gc.grayList == nil {
			break
		}

		obj := gc.grayList
		gc.grayList = obj.grayNext
		obj.grayNext = nil
		obj.color = gcBlack
		state := DisableInterrupts()
		gc.scanObjectAtomic(obj)
		EnableInterrupts(state)
	}

	gc.phase = gcSweep
	gc.sweepCurr = gc.head
	gc.sweepPrev = nil
	gcMu.Unlock()
}

// scanRangeAtomic scans a memory range without batching.
// PRECONDITION: interrupts are disabled.
func (gc *_gc) scanRangeAtomic(low, high uintptr) {
	for addr := low; addr < high; addr += gcWordSize {
		val := *(*uintptr)(unsafe.Pointer(addr))
		if obj := gc.findObject(val); obj != nil {
			gc.shade(obj)
		}
	}
}

// --------------------------------------------------------------------------
// Sweep phase -- incremental
// --------------------------------------------------------------------------

func (gc *_gc) sweep() {
	// Process objects one at a time, releasing lock between each.
	// This minimizes lock hold time and reduces interrupt latency.
	gcMu.Lock()
	curr := gc.sweepCurr
	prev := gc.sweepPrev

	if curr == nil {
		// Sweep complete
		gc.phase = gcIdle
		gcMu.Unlock()
		return
	}

	// Process a small batch
	count := 0
	for count < gcSweepBatch && curr != nil {
		next := curr.next

		if curr.color == gcWhite {
			// Object is unreachable, free it
			if prev == nil {
				gc.head = next
			} else {
				prev.next = next
			}
			gc.deregisterObject(curr)
			free(unsafe.Pointer(curr))

			// Don't update prev (it stays the same)
		} else {
			// Object survived, reset for next cycle
			curr.color = gcWhite
			curr.grayNext = nil
			prev = curr
		}

		count++
		curr = next
	}

	gc.sweepCurr = curr
	gc.sweepPrev = prev

	if gc.sweepCurr == nil {
		gc.phase = gcIdle
	}
	gcMu.Unlock()
}

// --------------------------------------------------------------------------
// Full GC -- stop-the-world
// --------------------------------------------------------------------------

// PRECONDITION: interrupts are disabled.
func (gc *_gc) fullGCLocked() {
	for obj := gc.head; obj != nil; obj = obj.next {
		obj.color = gcWhite
		obj.grayNext = nil
	}
	gc.grayList = nil

	gc.scanRangeAtomic(gcStackBottom(), gcStackTop())

	if headGoroutine != nil {
		g := headGoroutine
		for {
			low, high := gcGoroutineStack(g)
			gc.scanRangeAtomic(low, high)
			g = g.next
			if g == headGoroutine {
				break
			}
		}
	}

	gc.scanRangeAtomic(gcGlobalsStart(), gcGlobalsEnd())

	for gc.grayList != nil {
		obj := gc.grayList
		gc.grayList = obj.grayNext
		obj.grayNext = nil
		obj.color = gcBlack
		gc.scanObjectAtomic(obj)
	}

	prev := (*gcObject)(nil)
	curr := gc.head
	for curr != nil {
		next := curr.next
		if curr.color == gcWhite {
			if prev == nil {
				gc.head = next
			} else {
				prev.next = next
			}
			gc.deregisterObject(curr)
			free(unsafe.Pointer(curr))
		} else {
			curr.color = gcWhite
			curr.grayNext = nil
			prev = curr
		}
		curr = next
	}

	gc.grayList = nil
	gc.scanState = gcScanStack
	gc.currentAddress = 0
	gc.endAddress = 0
	gc.currentGoroutine = nil
	gc.sweepCurr = nil
	gc.sweepPrev = nil
	gc.phase = gcIdle
}

func (gc *_gc) fullGC() {
	gcMu.Lock()
	state := DisableInterrupts()
	gc.fullGCLocked()
	EnableInterrupts(state)
	gcMu.Unlock()
}

// --------------------------------------------------------------------------
// GC iteration -- called by gcmain each scheduler tick
// --------------------------------------------------------------------------

func (gc *_gc) iterate() {
	gcMu.Lock()
	phase := gc.phase
	gcMu.Unlock()

	switch phase {
	case gcIdle:
		// Start a new cycle.
		//
		// Set the phase first so any heap stores that happen after this point
		// use the write barrier.
		gcMu.Lock()
		gc.phase = gcMark
		gc.grayList = nil

		for obj := gc.head; obj != nil; obj = obj.next {
			obj.color = gcWhite
			obj.grayNext = nil
		}

		gc.scanState = gcScanStack
		gc.currentAddress = gcStackBottom()
		gc.endAddress = gcStackTop()
		gc.currentGoroutine = nil
		gc.sweepCurr = nil
		gc.sweepPrev = nil
		gcMu.Unlock()

	case gcMark:
		gc.markStep()

	case gcRemark:
		gc.remark()

	case gcSweep:
		gc.sweep()
	}
}

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

//go:export initgc runtime.initgc
func initgc() {
	gc.phase = gcIdle
	gc.grayList = nil
	gc.head = nil
	gc.scanState = gcScanStack
	gc.currentAddress = 0
	gc.endAddress = 0
	gc.currentGoroutine = nil
	gc.sweepCurr = nil
	gc.sweepPrev = nil

	gc.heapBase = uintptr(unsafe.Pointer(&__heap_start))
	heapEnd := uintptr(unsafe.Pointer(&__heap_end))
	gc.numSlots = (heapEnd - gc.heapBase) / gcPointerAlign
	gc.allocBitmap.init(gc.numSlots)
	gc.startBitmap.init(gc.numSlots)
}

//go:export alloc runtime.alloc
func alloc(size uintptr) unsafe.Pointer {
	gcMu.Lock()
	ptr := gcAllocLocked(size)
	gcMu.Unlock()
	return ptr
}

//go:nosplit
//go:nowritebarrier
func gcAllocLocked(size uintptr) unsafe.Pointer {
	size = (size + gcPointerAlign - 1) &^ (gcPointerAlign - 1)
	if size == 0 {
		size = gcPointerAlign
	}

	allocSize := gcObjectSize + size

	ptr := malloc(allocSize)
	if ptr == nil {
		gc.fullGCLocked()
		ptr = malloc(allocSize)
		if ptr == nil {
			abort()
		}
	}

	obj := (*gcObject)(ptr)
	obj.size = size

	// With an insertion barrier, new objects can be black.
	// Stores into them during mark/remark must go through the barrier.
	obj.color = gcBlack
	obj.grayNext = nil

	gc.registerObject(obj)

	obj.next = gc.head
	gc.head = obj

	return unsafe.Add(ptr, gcObjectSize)
}

//go:export gcmain runtime.gcmain
func gcmain() {
	for {
		gc.iterate()
		for range 10 {
			gosched()
		}
	}
}

// GC triggers a full garbage collection and blocks until complete.
func GC() {
	gc.fullGC()
}

// --------------------------------------------------------------------------
// Linker symbol accessors
// --------------------------------------------------------------------------

//go:inline
func gcStackTop() uintptr {
	return uintptr(unsafe.Pointer(&__stack_top))
}

//go:inline
func gcStackBottom() uintptr {
	return uintptr(unsafe.Pointer(&__stack_bottom))
}

//go:inline
func gcGlobalsStart() uintptr {
	return uintptr(unsafe.Pointer(&__gc_scan_start))
}

//go:inline
func gcGlobalsEnd() uintptr {
	return uintptr(unsafe.Pointer(&__gc_scan_end))
}

//go:inline
func gcGoroutineStack(g *goroutine) (low, high uintptr) {
	high = uintptr(unsafe.Add(g.stack, alignStack(goroutineStackSize)))
	low = uintptr(g.stackTop)
	if g == currentGoroutine {
		low = uintptr(currentStack())
	}
	return
}
