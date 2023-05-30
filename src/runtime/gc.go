package runtime

import "unsafe"

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
)

type object struct {
	addr   unsafe.Pointer
	next   *object
	size   uintptr
	marked bool
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
	disableInterrupts()
	heapLimit := (__heap_size / 100) * 70

	// Check if the GC needs to run
	allocSz := align(size + unsafe.Sizeof(object{}))
	if heapUsage+allocSz > heapLimit {
		// Garbage collect
		GC()

		// Check if there is room now
		if heapUsage+allocSz > __heap_size {
			panic("gc error: out of memory")
		}
	}

	// Allocate memory for the object
	obj := (*object)(malloc(unsafe.Sizeof(object{})))
	if obj == nil {
		panic("gc error: failed to allocate object")
	}

	obj.addr = malloc(size)
	if obj.addr == nil {
		panic("gc error: failed to allocate memory")
	}

	obj.size = size
	obj.next = head

	// Set the new object as the head
	head = obj
	numAllocas++

	// Update heapUsage
	info := mallinfo()
	heapUsage = info.uordblks

	// Return the address of the allocated memory
	enableInterrupts()
	return unsafe.Pointer(obj)
}

func freeObject(obj *object) {
	if obj != nil {
		free(obj.addr)
		free(unsafe.Pointer(obj))

		// Update heapUsage
		info := mallinfo()
		heapUsage = info.uordblks
	}
}

// markAll scans the stack from bottom to top looking for addresses that "look like a heap pointer".
func markAll() {
	dataStart := unsafe.Pointer(&__start_data)
	dataEnd := unsafe.Pointer(&__end_data)
	scan(dataStart, dataEnd)

	// Scan the goroutine stacks
	_task := headTask
	for {
		if _task != nil {
			scan(_task.stackTop, _task.stack)
			_task = _task.next
			if _task == headTask {
				break
			}
		}
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
				if !obj.marked {
					objAddr := uintptr(obj.addr)
					if addrVal == objAddr {
						// Mark this object
						obj.marked = true
					}
				}
				obj = obj.next
			}
		}
	}
}

func compact() {
	numAllocas = 0
	src := head
	var lastMarked *object

	for src != nil {
		next := src.next

		if src.marked {
			src.marked = false

			if lastMarked == nil {
				head = src
			} else {
				lastMarked.next = src
				if uintptr(lastMarked.addr)+uintptr(lastMarked.size) != uintptr(src.addr) {
					memmove(unsafe.Pointer(uintptr(lastMarked.addr)+uintptr(lastMarked.size)), src.addr, src.size)
				}
				src.addr = unsafe.Pointer(uintptr(lastMarked.addr) + uintptr(lastMarked.size))
			}

			lastMarked = src
		} else {
			freeObject(src)
		}

		src = next
	}

	if lastMarked != nil {
		lastMarked.next = nil
	} else {
		head = nil
	}
}

func align(n uintptr) uintptr {
	return n + (8 - (n % 8))
}

func GC() {
	markAll()
	compact()
}
