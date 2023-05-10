package runtime

import "unsafe"

//sigo:extern __stack_top __stack_top
//sigo:extern __heap_start __heap_start
//sigo:extern __heap_end __heap_end
//sigo:extern __heap_size __heap_size
//sigo:extern __start_data __start_data
//sigo:extern __end_data __end_data

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
	addr    unsafe.Pointer
	next    *object
	size    uintptr
	marked  bool
	padding [3]byte
}

func _malloc(size uintptr) unsafe.Pointer {
	ptr := malloc(size)
	if ptr != nil {
		heapUsage += align(size)
	}
	return ptr
}

func alloc(size uintptr) unsafe.Pointer {
	// Check if the GC needs to run
	allocSz := align(size + unsafe.Sizeof(object{}))
	if heapUsage+allocSz > __heap_size - 4096 {
		// Garbage collect
		GC()

		// Check if there is room now
		if heapUsage+allocSz > __heap_size {
			panic("gc error: out of memory")
		}
	}

	// Allocate memory for the object
	obj := (*object)(_malloc(unsafe.Sizeof(object{})))
	if obj == nil {
		panic("gc error: failed to allocate object")
	}

	obj.addr = _malloc(size)
	if obj.addr == nil {
		panic("gc error: failed to allocate memory")
	}

	obj.size = size
	obj.next = head

	// Set the new object as the head
	head = obj
	numAllocas++

	// Return the address of the allocated memory
	return unsafe.Pointer(obj)
}

func freeObject(obj *object) {
	if obj != nil {
		free(obj.addr)
		heapUsage -= align(obj.size)

		free(unsafe.Pointer(obj))
		heapUsage -= align(unsafe.Sizeof(object{}))
	}
}

// markAll scans the stack from bottom to top looking for addresses that "look like a heap pointer".
func markAll() {
	stackCurrent := currentStack()
	stackTop := unsafe.Pointer(&__stack_top)
	dataStart := unsafe.Pointer(&__start_data)
	dataEnd := unsafe.Pointer(&__end_data)

	scan(stackCurrent, stackTop)
	scan(dataStart, dataEnd)
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
	return n + (4 - (n % 4))
}

func GC() {
	markAll()
	compact()
}
