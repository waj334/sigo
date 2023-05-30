package runtime

import (
	"unsafe"
)

type taskState uint8

const (
	taskNotStarted taskState = iota
	taskIdle
	taskSleep
	taskRunning
)

type goroutine struct {
	fnPtr  unsafe.Pointer
	params unsafe.Pointer
}

type task struct {
	stackTop      unsafe.Pointer
	ctx           *goroutine
	stack         unsafe.Pointer
	next          *task
	prev          *task
	state         taskState
	sleepDeadline uint64
}

var (
	headTask   *task      = nil
	lastTask   *task      = nil
	timeSource TimeSource = &SysTickSource{}
)

func init() {
	*_goroutineStackSize = alignStack(*_goroutineStackSize)
}

//go:export currentTask runtime._currentTask
var currentTask *task = nil

//sigo:extern _goroutineStackSize runtime._goroutineStackSize
var _goroutineStackSize *uintptr

//go:linkname initTask runtime.initTask
func initTask(unsafe.Pointer)

//go:linkname alignStack runtime.alignStack
func alignStack(n uintptr) uintptr

//go:linkname _currentTick runtime.currentTick
func _currentTick() uint32

//go:linkname triggerPendSV _triggerPendSV
func triggerPendSV()

//go:export runScheduler runtime.runScheduler
func runScheduler() (shouldSwitch bool) {
	disableInterrupts()

	if headTask != nil {
		if currentTask == nil {
			// Initialize the current task
			currentTask = headTask
			lastTask = nil
		} else {
			if currentTask.state == taskRunning {
				// Move the current task to the idle state
				currentTask.state = taskIdle
			}

			// Update this task's stack pointer
			currentTask.stackTop = currentStack()

			// Switch to the next task
			lastTask = currentTask
			nextTask := lastTask.next

			for {
				if nextTask.state == taskSleep {
					currentTick := timeSource.Now()
					if currentTick > nextTask.sleepDeadline {
						nextTask.state = taskIdle
						nextTask.sleepDeadline = 0
					} else if nextTask == lastTask && nextTask.state == taskSleep {
						// All tasks are sleep. panic
						panic("all goroutines are sleep")
					} else {
						// Skip sleeping task
						nextTask = nextTask.next
						continue
					}
				}
				currentTask = nextTask
				break
			}
		}

		if currentTask != nil && currentTask != lastTask {
			switch currentTask.state {
			case taskNotStarted:
				// Initialize the stack for this task
				initTask(unsafe.Pointer(currentTask))

				// Change this task to the running state
				fallthrough
			case taskIdle:
				currentTask.state = taskRunning
				shouldSwitch = true
			}
		}
	}

	enableInterrupts()
	return
}

//go:export addTask runtime.addTask
func addTask(ptr unsafe.Pointer) {
	oldHead := headTask
	headTask = (*task)(malloc(unsafe.Sizeof(task{})))

	// Allocate stack for this goroutine
	headTask.stack = malloc(*_goroutineStackSize)
	headTask.ctx = (*goroutine)(ptr)
	headTask.stackTop = headTask.stack

	// Create a ring
	if oldHead == nil {
		headTask.next = headTask
		headTask.prev = headTask
	} else {
		// Insert the new task before the old head task
		headTask.next = oldHead
		headTask.prev = oldHead.prev

		oldHead.prev.next = headTask
		oldHead.prev = headTask
	}
}

//go:export runtime.removeTask
func removeTask(t *task) {
	// Free the stack memory
	free(t.stack)

	if t.next == t && t.prev == t {
		// There is only one task left.
		headTask = nil
		currentTask = nil
		lastTask = nil
	} else {
		// Remove the task from the ring.
		t.prev.next = t.next
		t.next.prev = t.prev

		// Advance to the next task
		if t == currentTask {
			currentTask = t.prev
		}

		// If the task being removed was the head task, set the next task as the new head task.
		if t == headTask {
			headTask = t.next
		}

		// If the task being removed was the current task, set the next task as the new current task.
		if t == currentTask {
			currentTask = t.next
		}

		// If the task being removed was the last task, set the previous task as the new last task.
		if t == lastTask {
			lastTask = t.prev
		}
	}

	// Free the task
	free(unsafe.Pointer(t))
}

//go:export _sleep runtime.sleep
func _sleep(d uint64) {
	if currentTask == nil {
		panic("sleep called from non-goroutine")
	}
	currentTask.sleepDeadline = timeSource.Now() + d
	currentTask.state = taskSleep

	// TODO: This needs to a less architecture-specific call
	triggerPendSV()
}
