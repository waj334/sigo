//go:build disabled

package runtime

import (
	"sync"
	"unsafe"
)

//sigo:extern __stack_top __stack_top
//sigo:extern __stack_bottom __stack_bottom
//sigo:extern __heap_start __heap_start
//sigo:extern __heap_end __heap_end
//sigo:extern __heap_size __heap_size
//sigo:extern __gc_scan_start __gc_scan_start
//sigo:extern __gc_scan_end __gc_scan_end
//sigo:extern mallinfo mallinfo

//go:export alloc runtime.alloc

const (
	blockSize = 4096
)

var (
	headObject     *object
	heapBuckets    *heapBucket
	heapLoadFactor uintptr
	numHeapBuckets uintptr
	heapUsage      uintptr
	numAllocas     uintptr
	maxAllocas     uintptr
	maxHeapSize    uintptr

	markBitmap     unsafe.Pointer
	markBitmapSize uintptr = 64

	__stack_top     unsafe.Pointer
	__stack_bottom  unsafe.Pointer
	__heap_start    unsafe.Pointer
	__heap_end      unsafe.Pointer
	__heap_size     uintptr
	__gc_scan_start unsafe.Pointer
	__gc_scan_end   unsafe.Pointer
	gcMu            sync.Mutex
)

type object struct {
	addr unsafe.Pointer
	next *object
	sz   uintptr
	//marked  bool
	//scanned bool
	_ [2]uint8 /* padding */
}

type heapBucket struct {
	head *heapBucketEntry
	next *heapBucket
}

type heapBucketEntry struct {
	obj  *object
	next *heapBucketEntry
}

//go:inline markIndex
func markIndex(obj *object) uintptr {
	heapStart := uintptr(unsafe.Pointer(&__heap_start))
	offset := uintptr(obj.addr) - heapStart
	// Divide by the size of a pointer to get an index
	return offset / unsafe.Sizeof(uintptr(0))
}

func resizeMarkBitmap(numWords uintptr) {
	if markBitmap != nil {
		// Expand or shrink the bitmap.
		markBitmap = realloc(markBitmap, numWords*unsafe.Sizeof(uintptr(0)))
	} else {
		// Allocate new memory for the bitmap.
		markBitmap = malloc(numWords * unsafe.Sizeof(uintptr(0)))
	}
	markBitmapSize = numWords
}

func ensureBitmapSize(index uintptr) {
	newSize := markBitmapSize
	for index >= newSize*unsafe.Sizeof(uintptr(0))*8 {
		newSize *= 2
	}

	if newSize > markBitmapSize {
		resizeMarkBitmap(newSize)
	}
}

//go:inline setMarkBit
func setMarkBit(index uintptr) {
	ensureBitmapSize(index)
	b := unsafe.Add(markBitmap, index/unsafe.Sizeof(uintptr(0)))
	*(*byte)(b) |= 1 << (index % unsafe.Sizeof(uintptr(0)))
}

//go:inline clearMarkBit
func clearMarkBit(index uintptr) {
	ensureBitmapSize(index)
	b := unsafe.Add(markBitmap, index/unsafe.Sizeof(uintptr(0)))
	*(*byte)(b) &^= 1 << (index % unsafe.Sizeof(uintptr(0)))
}

//go:inline isMarked
func isMarked(index uintptr) bool {
	ensureBitmapSize(index)
	b := unsafe.Add(markBitmap, index/unsafe.Sizeof(uintptr(0)))
	return *(*byte)(b)&(1<<(index%unsafe.Sizeof(uintptr(0)))) != 0
}

//go:export initgc runtime.initgc
func initgc() {
	heapLoadFactor = 4
	numHeapBuckets = 8
	maxAllocas = numHeapBuckets * heapLoadFactor
	maxHeapSize = blockSize * 2

	// Allocate buckets.
	heapBuckets = createBuckets(numHeapBuckets)
}

func createBuckets(n uintptr) *heapBucket {
	var head *heapBucket
	for i := uintptr(0); i < n; i++ {
		bucket := (*heapBucket)(malloc(unsafe.Sizeof(heapBucket{})))
		bucket.next = head
		head = bucket
	}

	return head
}

func resizeHeapBuckets() {
	// Determine whether to grow or shrink the number of buckets
	if numAllocas > maxAllocas {
		numHeapBuckets *= 2
	} else if numAllocas < (maxAllocas/4) && numHeapBuckets > 8 {
		numHeapBuckets /= 2
	} else {
		// If the number of allocations is within the acceptable range, don't resize
		return
	}

	// Allocate new buckets
	oldHeapBuckets := heapBuckets
	heapBuckets = createBuckets(numHeapBuckets)

	// Redistribute objects
	redistributeObjects(oldHeapBuckets)

	// Free the old buckets.
	freeBuckets(oldHeapBuckets)

	maxAllocas = numHeapBuckets * heapLoadFactor
}

func redistributeObjects(oldBuckets *heapBucket) {
	bucket := oldBuckets
	for bucket != nil {
		entry := bucket.head
		for entry != nil {
			nextEntry := entry.next
			hash := ptrHash(entry.obj.addr)
			newBucket := getBucket(heapBuckets, hash)
			entry.next = newBucket.head
			newBucket.head = entry
			entry = nextEntry
		}
		bucket = bucket.next
	}
}

func freeBuckets(head *heapBucket) {
	for head != nil {
		bucket := head
		head = head.next
		free(unsafe.Pointer(bucket))
	}
}

func getBucket(head *heapBucket, i uintptr) *heapBucket {
	if i >= 0 {
		bucket := head
		for ii := uintptr(0); bucket != nil; ii++ {
			if ii == i {
				return bucket
			}
			bucket = bucket.next
		}
	}
	return nil
}

func ptrHash(ptr unsafe.Pointer) uintptr {
	const shiftAmount = 3 // adjust based on your knowledge of the alignment
	shifted := uintptr(ptr) >> shiftAmount
	return shifted % numHeapBuckets
}

type strMallinfo struct {
	arena    uintptr /* Non-mmapped space allocated (bytes) */
	ordblks  uintptr /* Number of free chunks */
	smblks   uintptr /* Number of free fastbin blocks */
	hblks    uintptr /* Number of mmapped regions */
	hblkhd   uintptr /* Space allocated in mmapped regions (bytes) */
	usmblks  uintptr /* See below */
	fsmblks  uintptr /* Space in freed fastbin blocks (bytes) */
	uordblks uintptr /* Total allocated space (bytes) */
	fordblks uintptr /* Total free space (bytes) */
	keepcost uintptr /* Top-most, releasable space (bytes) */
}

func mallinfo() strMallinfo

func alloc(size uintptr) unsafe.Pointer {
	if size == 0 {
		return nil
	}

	// Align the size to the nearest word barrier
	size = align(size)

	// Lock the mutex before disabling the interrupt so that goroutines can
	// compete for the lock.
	gcMu.Lock()

	// Disable interrupts so that there is no context switch during memory
	// allocation.
	state := disableInterrupts()

	// Attempt to allocate memory for the object ref.
	objSize := unsafe.Sizeof(object{})
	ptr := malloc(objSize + size)
	if ptr == nil {
		// Heap is full. Perform a GC now to reclaim any unused memory.
		markAll()
		sweep()

		// Attempt to allocate again.
		ptr = malloc(size)
		if ptr == nil {
			gcMu.Unlock()
			enableInterrupts(state)

			// TODO: print the panic message
			// NOTE: Cannot panic normally here because panics require a heap allocation causing infinite recursion.
			//panic("gc error: out of memory")

			// Stop running.
			abort()
		}
	}

	// Set up the object ref.
	obj := (*object)(ptr)
	obj.addr = unsafe.Add(ptr, unsafe.Sizeof(object{}))
	obj.sz = size
	obj.next = headObject

	// Set this object reference as the new head.
	headObject = obj

	// Update bucket.
	hash := ptrHash(obj.addr)
	bucket := getBucket(heapBuckets, hash)
	entry := (*heapBucketEntry)(malloc(unsafe.Sizeof(heapBucketEntry{})))
	entry.obj = obj
	entry.next = bucket.head
	bucket.head = entry

	// Update metrics.
	numAllocas++
	heapUsage = mallinfo().uordblks

	// Unlock the mutex before enabling the interrupts to prevent a deadlock
	// that can occur if there is a context switch within this critical
	// section.
	gcMu.Unlock()
	enableInterrupts(state)

	// Return the starting address of the memory allocation
	return obj.addr
}

func freeObject(obj *object) {
	if obj != nil {
		// Remove from hash map
		var lastEntry *heapBucketEntry
		hash := ptrHash(obj.addr)
		bucket := getBucket(heapBuckets, hash)
		for entry := bucket.head; entry != nil; {
			if entry.obj == obj {
				if entry == bucket.head {
					bucket.head = entry.next
				} else {
					lastEntry.next = entry.next
				}

				// Free the memory for the removed entry
				free(unsafe.Pointer(entry))
			}
			// Advance
			lastEntry = entry
			entry = entry.next
		}

		memset(unsafe.Pointer(obj), 0, unsafe.Sizeof(object{}))
		free(unsafe.Pointer(obj))

		// Update metrics
		numAllocas--
		heapUsage = mallinfo().uordblks
	}
}

// markAll scans the stack from bottom to top looking for addresses that "look like a heap pointer".
func markAll() {
	// Scan the goroutine stacks
	_task := headTask
	for {
		if _task != nil {
			stackBottom := unsafe.Add(_task.stack, alignStack(goroutineStackSize))
			stackTop := _task.stackTop
			if _task == currentTask {
				// Do not miss any heap object in the current goroutine since it
				// will have a different stack pointer after when the context
				// switched to it.
				stackTop = currentStack()
			}
			scan(unsafe.Add(stackTop, -64), stackBottom)

			_task = _task.next
			if _task == headTask {
				break
			}
		}
	}

	// Scan the main stack
	mainStackTop := unsafe.Pointer(&__stack_top)
	mainStackBottom := unsafe.Pointer(&__stack_bottom)
	scan(mainStackBottom, mainStackTop)

	// Scan the memory region defined by the linker script. This region
	// should contain globals and such.
	start := unsafe.Pointer(&__gc_scan_start)
	end := unsafe.Pointer(&__gc_scan_end)
	scan(start, end)
}

func scan(start, end unsafe.Pointer) {
	heapStart := uintptr(unsafe.Pointer(&__heap_start))
	heapEnd := uintptr(unsafe.Pointer(&__heap_end))
	for ptr := start; uintptr(ptr) < uintptr(end); ptr = unsafe.Add(ptr, unsafe.Sizeof(uintptr(0))) {
		addrVal := *(*uintptr)(ptr)
		if addrVal >= heapStart && addrVal < heapEnd {
			// Look up the object storing this pointer in the hash map
			hash := ptrHash(unsafe.Pointer(addrVal))
			bucket := getBucket(heapBuckets, hash)
			entry := bucket.head
			for entry != nil {
				// Skip objects that are already marked
				if !isMarked(markIndex(entry.obj)) {
					objAddr := uintptr(entry.obj.addr)
					objEnd := uintptr(unsafe.Add(entry.obj.addr, entry.obj.sz))
					// Check if addrVal falls within the object's range
					if addrVal >= objAddr && addrVal < objEnd {
						// Mark this object
						setMarkBit(markIndex(entry.obj))
						//entry.obj.marked = true

						//if !entry.obj.scanned {
						//entry.obj.scanned = true
						scan(unsafe.Pointer(objAddr), unsafe.Pointer(objEnd))
						//}
					}
				}
				entry = entry.next
			}
		}
	}
}

func sweep() {
	var lastMarked *object
	it := headObject
	for it != nil {
		next := it.next
		if isMarked(markIndex(it)) {
			lastMarked = it
			//it.marked = false
			clearMarkBit(markIndex(it))
			//it.scanned = false
		} else {
			if it == headObject {
				// Set the next object as the new head
				headObject = next
			} else {
				// Remove this object from the linked list
				lastMarked.next = next
			}
			freeObject(it)
		}
		it = next
	}

	// Terminate the linked-list at the last object that was marked
	if lastMarked != nil {
		lastMarked.next = nil
	}

	// Update metrics
	heapUsage = mallinfo().uordblks
}

//go:inline align align
func align(n uintptr) uintptr {
	return (n + 8) - (n % 8)
}

func GC() {
	gcMu.Lock()
	state := disableInterrupts()
	markAll()
	sweep()
	gcMu.Unlock()
	enableInterrupts(state)
}

//go:export gcmain runtime.gcmain
func gcmain() {
	for {
		gcMu.Lock()
		state := disableInterrupts()
		if numAllocas > maxAllocas || heapUsage >= maxHeapSize {
			markAll()
			sweep()
			resizeHeapBuckets()

			// Grow the heap size to the nearest block size + 1
			maxHeapSize = (heapUsage + blockSize) - (heapUsage % blockSize)
			maxHeapSize += blockSize
		}

		gcMu.Unlock()
		enableInterrupts(state)
		schedulerPause()
	}
}
