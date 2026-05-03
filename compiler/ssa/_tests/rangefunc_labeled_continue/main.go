// Test: labeled `continue outer` inside a nested range-over-func body
// resumes the outer iter at its next yielded value, skipping the rest of
// the outer body.
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
	vSum       int
	tail       int
)

func runLabeledContinue() {
	outerIters = 0
	innerIters = 0
	vSum = 0
	tail = 0
outer:
	for v := range count3 {
		outerIters++
		vSum += v
		for w := range count3 {
			innerIters++
			if w == 1 {
				continue outer
			}
		}
		// This statement runs only if no `continue outer` fired during this v iteration.
		// In this test it never runs because every v reaches w==1.
		tail++
	}
}

func main() {
	common.Begin("rangefunc_labeled_continue")

	runLabeledContinue()

	// For each v in {0,1,2}: inner runs at w=0 (no continue), then w=1 fires
	// continue outer — inner iter stops. innerIters per v = 2. Total = 6.
	// outerIters = 3 (each v completes its outer iteration via continue).
	// tail = 0 (continue outer skips it every time).
	common.AssertEq("outerIters", outerIters, 3)
	common.AssertEq("innerIters", innerIters, 2*3)
	common.AssertEq("vSum", vSum, 0+1+2)
	common.AssertEq("tail", tail, 0)

	common.Done()
	common.Halt()
}
