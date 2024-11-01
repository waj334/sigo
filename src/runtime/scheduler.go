package runtime

import (
	"time"
	"unsafe"
)

type taskState uint8

const (
	taskNotStarted taskState = iota
	taskIdle
	taskSleep
	taskRunning
	taskPanicking
	taskRecovered
	taskWaiting
)

type _func struct {
	f    unsafe.Pointer
	args unsafe.Pointer
}

type task struct {
	stackTop      unsafe.Pointer
	__func        _func
	stack         unsafe.Pointer
	next          *task
	prev          *task
	state         taskState
	sleepDeadline uint64
	deferStack    *deferStack
	panicValue    any
}

//sigo:extern goroutineStackSize runtime._goroutineStackSize
//sigo:extern initTask runtime.initTask
//sigo:extern alignStack runtime.alignStack
//sigo:extern schedulerPause runtime.schedulerPause

//go:export lastTask runtime.lastTask
//go:export currentTask runtime.currentTask
//go:export runScheduler runtime.runScheduler
//go:export addTask runtime.addTask
//go:export removeTask runtime.removeTask
//go:export sleep runtime.sleep
//go:export waitTask runtime.waitTask
//go:export resumeTask runtime.resumeTask
//go:export runningTask runtime.runningTask

//sigo:required runScheduler

var (
	headTask           *task = nil
	lastTask           *task = nil
	currentTask        *task = nil
	goroutineStackSize uintptr
)

func initTask(unsafe.Pointer)
func alignStack(n uintptr) uintptr
func schedulerPause()

func runScheduler() (shouldSwitch bool) {
	state := disableInterrupts()

	// Check for stack overflow on current task
	if currentTask != nil {
		stackBottom := unsafe.Add(currentTask.stack, goroutineStackSize)
		stackSize := uintptr(stackBottom) - uintptr(currentStack())
		if stackSize > goroutineStackSize {
			panic("stack overflow")
		}
	}

	if headTask != nil {
		if currentTask == nil {
			// Initialize the current task
			currentTask = headTask
			lastTask = nil
		} else {
			if currentTask.state == taskRunning {
				// Move the current task to the idle state
				currentTask.state = taskIdle
			} else if currentTask.state == taskPanicking || currentTask.state == taskRecovered {
				// Do not allow any further context switches from this task
				// NOTE: Interrupts are intentionally not re-enabled. The panic will re-enable them if a panic is
				//		 recovered.
				lastTask = currentTask
				return
			}

			// Switch to the next task
			lastTask = currentTask
			nextTask := lastTask.next

			for {
				if nextTask.state == taskSleep {
					t := uint64(time.Now().UnixNano())
					if t > nextTask.sleepDeadline {
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
				} else if nextTask.state == taskWaiting {
					// skip waiting tasks
					nextTask = nextTask.next
					continue
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

	enableInterrupts(state)
	return
}

func addTask(f _func) {
	if f.f == nil {
		// Do nothing.
		return
	}

	state := disableInterrupts()
	oldHead := headTask

	// Allocate stack for this goroutine
	stackSize := goroutineStackSize
	stack := alloc(stackSize)

	// Create the new task
	headTask = &task{
		stack: stack,
		// initTask may move the top of stack pointer depending on the target machine's stack growth direction
		stackTop: stack,
		__func:   f,
		state:    taskNotStarted,
	}

	// Insert into ring
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
	enableInterrupts(state)
}

func removeTask(t *task) {
	state := disableInterrupts()

	// Free this task's stack
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

	enableInterrupts(state)
}

func waitTask(ptr unsafe.Pointer) {
	t := (*task)(ptr)
	if t.state != taskWaiting {
		state := disableInterrupts()
		t.state = taskWaiting
		enableInterrupts(state)

		// Schedule another task to begin running
		schedulerPause()
	}
}

func resumeTask(ptr unsafe.Pointer) {
	t := (*task)(ptr)
	if t.state == taskWaiting {
		state := disableInterrupts()
		t.state = taskIdle
		enableInterrupts(state)
	}
}

func runningTask() unsafe.Pointer {
	return unsafe.Pointer(currentTask)
}

func sleep(d uint64) {
	if currentTask == nil {
		panic("sleep called from non-goroutine")
	}
	currentTask.sleepDeadline = uint64(time.Now().UnixNano()) + d
	currentTask.state = taskSleep

	// Schedule another task to begin running
	schedulerPause()
}
