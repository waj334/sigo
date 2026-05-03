// Test: return from a void-returning function inside a range-over-func body.
// Exercises the "no result temps" path of Phase 2B (state slot only).
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func count(yield func(int) bool) {
	for i := 0; i < 10; i++ {
		if !yield(i) {
			return
		}
	}
}

var (
	calls int
	last  int
)

// returnVoid has no results. A bare `return` inside the range body should
// still propagate: state gets set to the sentinel, and the post-iter
// dispatch emits a void Return.
func returnVoid() {
	calls = 0
	last = -1
	for v := range count {
		calls++
		last = v
		if v == 2 {
			return
		}
	}
	// Marker: this line should NOT execute when return triggers above.
	last = 999
}

func main() {
	common.Begin("rangefunc_return_void")

	returnVoid()

	// After return inside body at v=2: calls=3 (yields 0,1,2), last=2,
	// and the trailing assignment to last=999 must NOT have run.
	common.AssertEq("calls", calls, 3)
	common.AssertEq("last", last, 2)

	common.Done()
	common.Halt()
}
