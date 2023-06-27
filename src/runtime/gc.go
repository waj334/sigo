package runtime

import (
	"sync"
	"unsafe"
)

//sigo:extern __stack_top __stack_top
//sigo:extern __heap_start __heap_start
//sigo:extern __heap_end __heap_end
//sigo:extern __heap_size __heap_size
//sigo:extern __start_data __start_data
//sigo:extern __end_data __end_data
//sigo:extern mallinfo mallinfo

var (
	head       *object
	heapUsage  uintptr
	numAllocas uintptr

	__stack_top  unsafe.Pointer
	__heap_start unsafe.Pointer
	__heap_end   unsafe.Pointer
	__heap_size  uintptr

	__start_data unsafe.Pointer
	__end_data   unsafe.Pointer
	gcMu         sync.Mutex
)

type object struct {
	addr unsafe.Pointer
	next *object
	sz   uintptr
}

func (o *object) mark() {
	o.sz |= 1 << 31
}

func (o *object) unmark() {
	o.sz &= ^uintptr(1 << 31)
}

func (o *object) isMarked() bool {
	return (o.sz >> 31) != 0
}

func (o *object) size() uintptr {
	return o.sz & ^uintptr(1<<31)
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

	// Lock the mutex before disabling the interrupt so that goroutines can
	// compete for the lock.
	gcMu.Lock()

	// Disable interrupts so that there is no context switch during memory
	// allocation.
	state := disableInterrupts()

	// Attempt to allocate memory for the object ref
	ptr := malloc(align(unsafe.Sizeof(object{})) + size)
	if ptr == nil {
		// Heap is full. Perform a GC now to reclaim any unused memory
		markAll()
		sweep()

		// Attempt to allocate again
		ptr = malloc(size)
		if ptr == nil {
			gcMu.Unlock()
			panic("gc error: out of memory")
		}
	}

	// Set up the object ref
	obj := (*object)(ptr)
	obj.addr = unsafe.Add(ptr, align(unsafe.Sizeof(object{})))
	obj.sz = size
	obj.next = head

	// Set this object reference as the new head
	head = obj

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
		free(unsafe.Pointer(obj))

		// Update metrics
		heapUsage = mallinfo().uordblks
	}
}

// markAll scans the stack from bottom to top looking for addresses that "look like a heap pointer".
func markAll() {
	// Scan the globals
	dataStart := unsafe.Pointer(&__start_data)
	dataEnd := unsafe.Pointer(&__end_data)
	scan(dataStart, dataEnd)

	// Scan the goroutine stacks
	_task := headTask
	for {
		if _task != nil {
			stackBottom := unsafe.Add(_task.stack, alignStack(*goroutineStackSize))
			scan(currentStack(), stackBottom)

			_task = _task.next
			if _task == headTask {
				break
			}
		}
	}

	// Scan current heap objects
	obj := head
	for obj != nil {
		scan(obj.addr, unsafe.Add(obj.addr, obj.size()))
		obj = obj.next
	}
}

func scan(start, end unsafe.Pointer) {
	heapStart := uintptr(unsafe.Pointer(&__heap_start))
	heapEnd := uintptr(unsafe.Pointer(&__heap_end))
	for ptr := start; uintptr(ptr) < uintptr(end); ptr = unsafe.Add(ptr, unsafe.Sizeof(uintptr(0))) {
		addrVal := *(*uintptr)(ptr)
		if addrVal >= heapStart && addrVal < heapEnd {
			// Look up the object storing this pointer
			obj := head
			for obj != nil {
				// Skip objects that are already marked
				if !obj.isMarked() {
					objAddr := uintptr(obj.addr)
					if addrVal == objAddr {
						// Mark this object
						obj.mark()
					}
				}
				obj = obj.next
			}
		}
	}
}

func sweep() {
	var lastMarked *object
	it := head
	for it != nil {
		next := it.next
		if it.isMarked() {
			lastMarked = it
			it.unmark()
		} else {
			if it == head {
				// Set the next object as the new head
				head = next
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
	numAllocas = 0
	heapUsage = mallinfo().uordblks
}

func align(n uintptr) uintptr {
	return n + (unsafe.Sizeof(uintptr(0)) - (n % unsafe.Sizeof(uintptr(0))))
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
	heapLimit := (__heap_size / 100) * 70
	for {
		gcMu.Lock()
		state := disableInterrupts()
		if numAllocas > 10 || heapUsage >= heapLimit {
			markAll()
			sweep()
		}
		gcMu.Unlock()
		enableInterrupts(state)
		schedulerPause()
	}
}
