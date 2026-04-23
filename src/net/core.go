package net

import (
	"runtime"
	"time"
)

const (
	timerServiceInterval = 250 * time.Millisecond
)

var (
	opChan chan operation
)

type operation struct {
	fn   func()
	done chan struct{}
}

func init() {
	opChan = make(chan operation, 1)

	// Init the LWIP stack.
	lwipInit()
}

func queueOperation(fn func()) {
	op := operation{
		fn:   fn,
		done: make(chan struct{}, 1),
	}

	// Submit the operation to the queue.
	opChan <- op

	// Park this goroutine until this operation is processed.
	<-op.done
}

func Poll() {
	ticker := time.NewTicker(timerServiceInterval)
	for {
		// Service pending operations before anything else.
		select {
		case op := <-opChan:
			op.fn()
			op.done <- struct{}{}
			continue
		default:
		}

		// Service lwip timers on each tick.
		select {
		case <-ticker.C:
			lwipSysCheckTimeouts()
		default:
			// Drain one frame per interface per iteration so that opChan and the
			// timer check are visited between every frame (feedFrame can block via
			// goNetifLinkoutput → SendEthernet waiting for TX credits).
			for _, ni := range interfaces {
				select {
				case frame := <-ni.rx:
					ni.feedFrame(frame)
				default:
				}
			}

			// Yield to other goroutines.
			runtime.Gosched()
		}
	}
}
