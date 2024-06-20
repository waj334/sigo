package runtime

import (
	"sync"
	"unsafe"
)

//sigo:extern __gc_scan_start __gc_scan_start
//sigo:extern __gc_scan_end __gc_scan_end
//sigo:extern __heap_start __heap_start
//sigo:extern __heap_end __heap_end
//sigo:extern __stack_top __stack_top
//sigo:extern __stack_bottom __stack_bottom

const (
	initialStorageSize = 32
)

var (
	_gc gc

	__gc_scan_start unsafe.Pointer
	__gc_scan_end   unsafe.Pointer
	__heap_start    unsafe.Pointer
	__heap_end      unsafe.Pointer
	__stack_top     unsafe.Pointer
	__stack_bottom  unsafe.Pointer
)

type color uint8

const (
	white color = iota
	gray  color = iota
	black color = iota
)

type gc struct {
	head    *heapObject
	static  []*heapObject
	dynamic []*heapObject
	gray    []*heapObject
	mutex   sync.Mutex

	heapStart uintptr
	heapEnd   uintptr
}

func (*gc) allocSlice(s []*heapObject, n int) []*heapObject {
	arr := unsafe.Pointer(unsafe.SliceData(s))
	newSlice := _slice{
		array: realloc(arr, unsafe.Sizeof(&heapObject{})*uintptr(n)),
		len:   len(s),
		cap:   n,
	}
	return *(*[]*heapObject)(unsafe.Pointer(&newSlice))
}

func (g *gc) addRoot(s *[]*heapObject, obj *heapObject) {
	_s := *s
	if len(_s) == 0 {
		*s = g.allocSlice(_s, initialStorageSize)
	} else if len(_s)+1 > cap(_s) {
		// Reallocate the backing array for the slice.
		*s = g.allocSlice(_s, cap(_s)*2)
	}
	*s = append(*s, obj)
}

//go:export initgc runtime.initgc
func initgc() {
	_gc.heapStart = uintptr(unsafe.Pointer(&__heap_start))
	_gc.heapEnd = uintptr(unsafe.Pointer(&__heap_end))
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

func (h *heapObject) setColor(col color) {
	h.data &^= (h.data &^ 0xF000_0000) | (1 << (28 + col))
}

func (h *heapObject) color() color {
	return color((h.data & 0xF000_0000) >> 28)
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
		// TODO: Attempt to reclaim memory.
		//_gc.run()

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

func scan(start unsafe.Pointer, end unsafe.Pointer) {
	for ptr := uintptr(start); ptr < uintptr(end); ptr += unsafe.Sizeof(uintptr(0)) {
		val := *(*uintptr)(unsafe.Pointer(ptr))
		if isHeapPointer(val) {

		}

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

// go:inline isHeapPointer
func isHeapPointer(val uintptr) bool {
	return val >= _gc.heapStart && val < _gc.heapEnd
}
