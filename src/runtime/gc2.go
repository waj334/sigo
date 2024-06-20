//go:build disabled

package runtime

import (
	"sync"
	"unsafe"
)

//sigo:extern __stack_top __stack_top
//sigo:extern __stack_bottom __stack_bottom
//sigo:extern __gc_scan_start __gc_scan_start
//sigo:extern __gc_scan_end __gc_scan_end

var (
	_gc gc

	__stack_top     unsafe.Pointer
	__stack_bottom  unsafe.Pointer
	__gc_scan_start unsafe.Pointer
	__gc_scan_end   unsafe.Pointer
)

type gc struct {
	head  *heapObject
	mutex sync.Mutex
}

func (*gc) run() {
	mark()
	sweep()
}

type heapObject struct {
	ptr  unsafe.Pointer
	next *heapObject
	data uint32
}

func (h *heapObject) setSize(size uintptr) {
	h.data = (h.data &^ 0xFFF_FFFF) | (uint32(size) & 0xFFF_FFFF)
}

func (h *heapObject) size() uintptr {
	return uintptr(h.data & 0xFFF_FFFF)
}

func (h *heapObject) setMarked(marked bool) {
	if marked {
		h.data |= 1 << 28
	} else {
		h.data &^= 1 << 28
	}
}

func (h *heapObject) marked() bool {
	return h.data&(1<<28) != 0
}

func (h *heapObject) setScanned(scanned bool) {
	if scanned {
		h.data |= 1 << 29
	} else {
		h.data &^= 1 << 29
	}
}

func (h *heapObject) scanned() bool {
	return h.data&(1<<29) != 0
}

//go:export alloc runtime.alloc
func alloc(size uintptr) unsafe.Pointer {
	_gc.mutex.Lock()
	state := disableInterrupts()

	if size == 0 {
		_gc.mutex.Unlock()
		enableInterrupts(state)
		return nil
	}

	// Allocate memory for the object on the heap.
	ptr := malloc(size + unsafe.Sizeof(heapObject{}))
	if ptr == nil {
		// Attempt to reclaim memory.
		_gc.run()

		// Retry the allocation.
		ptr = malloc(size + unsafe.Sizeof(heapObject{}))

		if ptr == nil {
			// There is no free memory.
			abort()
		}
	}

	// Initialize the heap object.
	obj := (*heapObject)(ptr)
	obj.ptr = unsafe.Add(ptr, unsafe.Sizeof(heapObject{}))
	obj.setSize(size)

	// Insert into the linked list of heap objects.
	obj.next = _gc.head
	_gc.head = obj

	_gc.mutex.Unlock()
	enableInterrupts(state)
	return obj.ptr
}

func scan(start unsafe.Pointer, size uintptr) {
	for offset := uintptr(0); offset < size; offset += unsafe.Sizeof(uintptr(0)) {
		ptr := *(*uintptr)(unsafe.Add(start, offset))

		// Find a heap object representing this pointer value.
		for obj := _gc.head; obj != nil; obj = obj.next {
			// Skip marked objects.
			if obj.marked() {
				continue
			}

			// Does the pointer value match this object?
			if uintptr(obj.ptr) == ptr {
				// Mark this object.
				obj.setMarked(true)
				if !obj.scanned() {
					// Scan this root object.
					obj.setScanned(true)
					scan(obj.ptr, obj.size())
				}
			}
		}
	}
}

func mark() {
	// Scan all goroutine stacks.
	started := false
	for g := headTask; g != nil; g = g.next {
		// Determine the top of the stack.
		var top unsafe.Pointer
		if g == currentTask {
			// Get the pointer to the top of the current stack.
			// NOTE: This is necessary because the stack pointer is not stored until a context switch.
			top = currentStack()
		} else {
			top = g.stackTop
		}

		// Calculate the amount of stack memory used by this goroutine.
		bottom := unsafe.Add(g.stack, alignStack(goroutineStackSize))
		size := uintptr(bottom) - uintptr(top)

		// Scan the stack memory top to bottom.
		scan(top, size)

		// NOTE: The task list is circular.
		if started && g == headTask {
			break
		}
		started = true
	}

	// Scan the main stack top to bottom.
	size := uintptr(unsafe.Pointer(&__stack_top)) - uintptr(unsafe.Pointer(&__stack_bottom))
	scan(unsafe.Pointer(&__stack_top), size)

	// Scan the globals and such.
	size = uintptr(unsafe.Pointer(&__gc_scan_end)) - uintptr(unsafe.Pointer(&__gc_scan_start))
	scan(unsafe.Pointer(&__gc_scan_start), size)
}

func sweep() {
	var lastObj *heapObject
	for obj := _gc.head; obj != nil; {
		if !obj.marked() {
			// Cache the next object to be swept.
			next := obj.next

			// Remove this object from the linked list.
			if lastObj != nil {
				lastObj.next = obj.next
			} else {
				_gc.head = obj.next
			}

			// Free this object.
			free(unsafe.Pointer(obj))

			// Advance.
			obj = next
		} else {
			// Reset state bits.
			obj.setMarked(false)
			obj.setScanned(false)
		}
		lastObj = obj
	}
}

func GC() {
	_gc.mutex.Lock()
	state := disableInterrupts()
	_gc.run()
	_gc.mutex.Unlock()
	enableInterrupts(state)
}

//go:export initgc runtime.initgc
func initgc() {

}

//go:export gcmain runtime.gcmain
func gcmain() {
	for {
		_gc.mutex.Lock()
		state := disableInterrupts()
		_gc.run()
		_gc.mutex.Unlock()
		enableInterrupts(state)
	}
}
