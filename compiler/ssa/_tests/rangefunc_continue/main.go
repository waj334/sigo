// Test: continue inside a range-over-func body.
//
// Verifies that `continue` skips the rest of the body and proceeds to the
// next yielded value (i.e. branches to the closure's fallthrough block,
// which returns true).
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func count(yield func(int) bool) {
	for i := 0; i < 6; i++ {
		if !yield(i) {
			return
		}
	}
}

func main() {
	common.Begin("rangefunc_continue")

	// Sum only odd values; even values are skipped via continue.
	sum := 0
	calls := 0
	postIncrement := 0
	for v := range count {
		calls++
		if v%2 == 0 {
			continue
		}
		sum += v
		postIncrement++
	}

	common.AssertEq("calls", calls, 6)                 // yielded all of 0..5
	common.AssertEq("sum", sum, 1+3+5)                 // odd values only
	common.AssertEq("postIncrement", postIncrement, 3) // body completed 3 times

	common.Done()
	common.Halt()
}
