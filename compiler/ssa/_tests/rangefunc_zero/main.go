// Test: range-over-func with a zero-arg Seq0.
//
// Iterates a `func(yield func() bool)` and verifies that the body executes
// once per yield, with no iteration variable.
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func ticks(yield func() bool) {
	for i := 0; i < 7; i++ {
		if !yield() {
			return
		}
	}
}

func main() {
	common.Begin("rangefunc_zero")

	calls := 0
	for range ticks {
		calls++
	}

	common.AssertEq("calls", calls, 7)

	common.Done()
	common.Halt()
}
