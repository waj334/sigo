package runtime

import (
	"sync"
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

	gc _gc
)

const (
	gcWordSize      = unsafe.Sizeof(uintptr(0))
	gcPointerAlign  = unsafe.Alignof(uintptr(0))
	gcObjectSize    = unsafe.Sizeof(gcObject{})
	gcMaxIterations = 100
)

type gcColor uint8

const (
	gcWhite gcColor = iota
	gcGray
	gcBlack
)

type gcPhase uint8

const (
	gcIdle gcPhase = iota
	gcMark
	gcSweep
)

type gcScanState uint8

const (
	gcScanStack gcScanState = iota
	gcScanGoroutines
	gcScanGlobals
	gcScanGray
)

type gcObject struct {
	next  *gcObject
	size  uintptr
	color gcColor
}

type _gc struct {
	head             *gcObject
	toScan           *gcObject
	currentAddress   uintptr
	endAddress       uintptr
	currentGoroutine *task
	mutex            sync.Mutex
	phase            gcPhase
	scanState        gcScanState
}

func (gc *_gc) fullGC() {
	// Reset all objects to white.
	for obj := gc.head; obj != nil; obj = obj.next {
		obj.color = gcWhite
	}

	gc.startMark()
	for gc.phase != gcIdle {
		gc.iterate()
	}
}

func (gc *_gc) startMark() {
	// Prepare for marking phase.
	gc.scanState = gcScanStack
	gc.currentAddress = gcStackTop()
	gc.endAddress = gcStackBottom()
	gc.phase = gcMark
}

func (gc *_gc) mark() {
	switch gc.scanState {
	case gcScanStack, gcScanGoroutines, gcScanGlobals:
		gc.markRoots()
	case gcScanGray:
		gc.markGrayObjects()
	}
}

func (gc *_gc) markRoots() {
	for i := 0; i < gcMaxIterations && gc.currentAddress < gc.endAddress; i++ {
		ptr := *(*unsafe.Pointer)(unsafe.Pointer(gc.currentAddress))
		if obj := gc.findObject(uintptr(ptr)); obj != nil {
			state := disableInterrupts()
			obj.color = gcGray
			enableInterrupts(state)
		}
		gc.currentAddress += gcWordSize
	}

	if gc.currentAddress >= gc.endAddress {
		gc.moveToNextScanState()
	}
}

func (gc *_gc) markGrayObjects() {
	for i := 0; i < gcMaxIterations && gc.toScan != nil; i++ {
		if gc.toScan.color == gcGray {
			state := disableInterrupts()
			gc.toScan.color = gcBlack
			enableInterrupts(state)
			gc.scanObject(gc.toScan)
		}
		gc.toScan = gc.toScan.next
	}

	if gc.toScan == nil {
		gc.phase = gcSweep
		gc.toScan = gc.head
	}
}

func (gc *_gc) sweep() {
	state := disableInterrupts()

	var prev *gcObject
	iteration := 0
	curr := gc.toScan

	for iteration < gcMaxIterations && curr != nil {
		next := curr.next

		if curr.color == gcWhite {
			if prev == nil {
				gc.head = next
			} else {
				prev.next = next
			}
			free(unsafe.Pointer(curr))
		} else {
			curr.color = gcWhite
			prev = curr
		}
		iteration++
		curr = next
	}

	gc.toScan = curr
	if gc.toScan == nil {
		gc.phase = gcIdle
	}
	enableInterrupts(state)
}

func (gc *_gc) findObject(val uintptr) *gcObject {
	if !isHeapPointer(val) {
		return nil
	}
	for obj := gc.head; obj != nil; obj = obj.next {
		objPtr := uintptr(unsafe.Pointer(obj))
		dataPtr := objPtr + gcObjectSize
		if val >= dataPtr && val < dataPtr+obj.size {
			return obj
		}
	}
	return nil
}

func (gc *_gc) scanObject(obj *gcObject) {
	dataPtr := uintptr(unsafe.Pointer(obj)) + gcObjectSize
	for ptr := dataPtr; ptr < dataPtr+obj.size; ptr += gcWordSize {
		childPtr := *(*unsafe.Pointer)(unsafe.Pointer(ptr))
		if child := gc.findObject(uintptr(childPtr)); child != nil {
			if child.color == gcWhite {
				state := disableInterrupts()
				child.color = gcGray
				enableInterrupts(state)
			}
		}
	}
}

func (gc *_gc) iterate() {
	switch gc.phase {
	case gcMark:
		gc.mark()
	case gcSweep:
		gc.sweep()
	default:
		// Do nothing
	}
}

func (gc *_gc) moveToNextScanState() {
	switch gc.scanState {
	case gcScanStack:
		gc.scanState = gcScanGoroutines
		if headTask != nil {
			gc.currentAddress, gc.endAddress = gcGoroutineStack(headTask)
			gc.currentGoroutine = headTask
		} else {
			gc.moveToNextScanState()
		}
	case gcScanGoroutines:
		if gc.currentGoroutine.next != headTask {
			gc.currentAddress, gc.endAddress = gcGoroutineStack(gc.currentGoroutine.next)
			gc.currentGoroutine = gc.currentGoroutine.next
		} else {
			gc.scanState = gcScanGlobals
			gc.currentAddress = gcGlobalsStart()
			gc.endAddress = gcGlobalsEnd()
		}
	case gcScanGlobals:
		gc.scanState = gcScanGray
		gc.toScan = gc.head
	case gcScanGray:
		// This case should be unreachable. Abort.
		abort()
	}
}

//go:export initgc runtime.initgc
func initgc() {
	gc.phase = gcIdle
}

//go:export alloc runtime.alloc
func alloc(size uintptr) unsafe.Pointer {
	gc.mutex.Lock()

	allocSize := gcObjectSize + size

	state := disableInterrupts()
	ptr := malloc(allocSize)
	enableInterrupts(state)

	if ptr == nil {
		// Attempt to reclaim memory now.
		gc.fullGC()

		state = disableInterrupts()
		ptr = malloc(allocSize)
		enableInterrupts(state)

		if ptr == nil {
			gc.mutex.Unlock()
			abort()
		}
	}

	state = disableInterrupts()
	obj := (*gcObject)(ptr)
	obj.next = gc.head
	// NOTE: Objects are born black to prevent sweeping them early.
	obj.color = gcBlack
	obj.size = size
	gc.head = obj
	enableInterrupts(state)

	if gc.phase == gcIdle {
		// Transition to mark phase.
		gc.startMark()
	}

	gc.mutex.Unlock()
	return unsafe.Add(ptr, gcObjectSize)
}

//go:export gcmain runtime.gcmain
func gcmain() {
	for {
		gc.mutex.Lock()
		gc.iterate()
		gc.mutex.Unlock()
		schedulerPause()
	}
}

func GC() {
	gc.mutex.Lock()
	gc.fullGC()
	gc.mutex.Unlock()
}

// go:inline isHeapPointer
func isHeapPointer(val uintptr) bool {
	return val >= gcHeapStart() && val < gcHeapEnd()
}

//go:inline gcHeapStart
func gcHeapStart() uintptr {
	return uintptr(unsafe.Pointer(&__heap_start))
}

//go:inline gcHeapEnd
func gcHeapEnd() uintptr {
	return uintptr(unsafe.Pointer(&__heap_end))
}

//go:inline gcStackTop
func gcStackTop() uintptr {
	return uintptr(unsafe.Pointer(&__stack_top))
}

//go:inline gcStackBottom
func gcStackBottom() uintptr {
	return uintptr(unsafe.Pointer(&__stack_bottom))
}

//go:inline gcGlobalsStart
func gcGlobalsStart() uintptr {
	return uintptr(unsafe.Pointer(&__gc_scan_start))
}

//go:inline gcGlobalsEnd
func gcGlobalsEnd() uintptr {
	return uintptr(unsafe.Pointer(&__gc_scan_end))
}

//go:inline gcGoroutineStack
func gcGoroutineStack(t *task) (top, bottom uintptr) {
	bottom = uintptr(unsafe.Add(t.stack, alignStack(goroutineStackSize)))
	top = uintptr(t.stackTop)
	if t == currentTask {
		// Do not miss any heap object in the current goroutine since it
		// will have a different stack pointer after when the context
		// switched to it.
		top = uintptr(currentStack())
	}

	// TODO: Remember why this was needed and derive the value of the constant from the current architecture.
	//top -= 64
	return
}
