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

const (
	blockSize = 4096
)

var (
	headObject     *object
	heapBuckets    *heapBucket
	heapLoadFactor uintptr
	numBuckets     uintptr
	heapUsage      uintptr
	numAllocas     uintptr
	maxAllocas     uintptr
	maxHeapSize    uintptr

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
	addr    unsafe.Pointer
	next    *object
	sz      uintptr
	marked  bool
	scanned bool
	_       [2]uint8 /* padding */
}

type heapBucket struct {
	head *heapBucketEntry
	next *heapBucket
}

type heapBucketEntry struct {
	obj  *object
	next *heapBucketEntry
}

//go:export initgc runtime.initgc
func initgc() {
	heapLoadFactor = 4
	numBuckets = 8
	maxAllocas = numBuckets * heapLoadFactor
	maxHeapSize = blockSize * 2

	// Allocate new buckets
	for i := uintptr(0); i < numBuckets; i++ {
		bucket := (*heapBucket)(malloc(unsafe.Sizeof(heapBucket{})))
		bucket.next = heapBuckets
		heapBuckets = bucket
	}
}

func resizeHeapBuckets() {
	// Determine whether to grow or shrink the number of buckets
	if numAllocas > maxAllocas {
		numBuckets *= 2
	} else if numAllocas < (maxAllocas/4) && numBuckets > 8 {
		numBuckets /= 2
	} else {
		// If the number of allocations is within the acceptable range, don't resize
		return
	}

	oldHeapBuckets := heapBuckets
	heapBuckets = nil

	// Allocate new buckets
	for i := uintptr(0); i < numBuckets; i++ {
		bucket := (*heapBucket)(malloc(unsafe.Sizeof(heapBucket{})))
		bucket.next = heapBuckets
		heapBuckets = bucket
	}

	// Redistribute objects
	bucket := oldHeapBuckets
	for bucket != nil {
		entry := bucket.head
		for entry != nil {
			// Copy the entry
			ptr := malloc(unsafe.Sizeof(heapBucketEntry{}))
			memcpy(ptr, unsafe.Pointer(entry), unsafe.Sizeof(heapBucketEntry{}))
			newEntry := (*heapBucketEntry)(ptr)

			// Hash the address
			ii := ptrHash(entry.obj.addr)

			// Locate the bucket to place this entry
			newBucket := getBucket(heapBuckets, ii)

			//Prepend the entry
			newEntry.next = newBucket.head
			newBucket.head = newEntry

			// Advance
			lastEntry := entry
			entry = entry.next

			// Free the last entry
			if lastEntry != nil {
				free(unsafe.Pointer(lastEntry))
			}
		}

		// Advance
		lastBucket := bucket
		bucket = bucket.next

		// Free the last bucket
		if lastBucket != nil {
			free(unsafe.Pointer(lastBucket))
		}
	}

	maxAllocas = numBuckets * heapLoadFactor
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
	return shifted % numBuckets
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

	// Attempt to allocate memory for the object ref
	ptr := malloc(unsafe.Sizeof(object{}) + size)
	if ptr == nil {
		// Heap is full. Perform a GC now to reclaim any unused memory
		markAll()
		sweep()

		// Attempt to allocate again
		ptr = malloc(size)
		if ptr == nil {
			gcMu.Unlock()
			enableInterrupts(state)
			panic("gc error: out of memory")
		}
	}

	// Set up the object ref
	obj := (*object)(ptr)
	obj.addr = unsafe.Add(ptr, unsafe.Sizeof(object{}))
	obj.sz = size
	obj.next = headObject

	// Set this object reference as the new head
	headObject = obj

	// Update bucket
	hash := ptrHash(obj.addr)
	bucket := getBucket(heapBuckets, hash)
	entry := (*heapBucketEntry)(malloc(unsafe.Sizeof(heapBucketEntry{})))
	entry.obj = obj
	entry.next = bucket.head
	bucket.head = entry

	// Update metrics
	numAllocas++
	heapUsage = mallinfo().uordblks

	// Unlock the mutex before enabling the interrupts to prevent a deadlock
	// that can occur if there is a context switch within this critical
	// section.
	gcMu.Unlock()

	// Allow context switches not
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
			stackBottom := unsafe.Add(_task.stack, alignStack(*goroutineStackSize))
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
				if !entry.obj.marked {
					objAddr := uintptr(entry.obj.addr)
					objEnd := uintptr(unsafe.Add(entry.obj.addr, entry.obj.sz))
					// Check if addrVal falls within the object's range
					if addrVal >= objAddr && addrVal < objEnd {
						// Mark this object
						entry.obj.marked = true

						if !entry.obj.scanned {
							entry.obj.scanned = true
							scan(unsafe.Pointer(objAddr), unsafe.Pointer(objEnd))
						}
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
		if it.marked {
			lastMarked = it
			it.marked = false
			it.scanned = false
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
