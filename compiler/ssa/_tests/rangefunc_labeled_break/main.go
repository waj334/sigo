// Test: labeled `break outer` inside a nested range-over-func body exits
// the outer iter and falls through past the outer for-loop.
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func count3(yield func(int) bool) {
	for i := 0; i < 3; i++ {
		if !yield(i) {
			return
		}
	}
}

var (
	outerIters int
	innerIters int
	postLoop   int
)

func runLabeledBreak() {
	outerIters = 0
	innerIters = 0
	postLoop = 0
outer:
	for v := range count3 {
		outerIters++
		for w := range count3 {
			innerIters++
			if v == 1 && w == 1 {
				break outer
			}
		}
	}
	postLoop = 1
}

func main() {
	common.Begin("rangefunc_labeled_break")

	runLabeledBreak()

	// v=0: inner runs 3 times (w=0,1,2). After inner loop, outer continues.
	// v=1: inner runs 2 times (w=0,1). At w=1, break outer fires.
	// outer breaks out. postLoop runs.
	common.AssertEq("outerIters", outerIters, 2)
	common.AssertEq("innerIters", innerIters, 3+2)
	common.AssertEq("postLoop", postLoop, 1)

	common.Done()
	common.Halt()
}
