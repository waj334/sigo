// Test: range-over-func with a single-value Seq.
//
// Iterates a `func(yield func(int) bool)` and verifies that all yielded
// values reach the loop body.
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func count(yield func(int) bool) {
	for i := 0; i < 5; i++ {
		if !yield(i) {
			return
		}
	}
}

func main() {
	common.Begin("rangefunc_basic")

	sum := 0
	calls := 0
	for v := range count {
		sum += v
		calls++
	}

	common.AssertEq("calls", calls, 5)
	common.AssertEq("sum", sum, 0+1+2+3+4)

	common.Done()
	common.Halt()
}
