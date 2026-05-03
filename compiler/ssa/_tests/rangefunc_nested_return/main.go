// Test: return inside a nested range-over-func body propagates through both
// iter calls to the outermost enclosing function.
//
// The body of the inner closure stores results into temps captured from the
// outermost function via FreeVar indirection. After the inner iter call,
// the inner post-iter dispatch sees state == returnSentinel and emits
// `return false` from the outer closure (propagation, not the actual outer
// Return). The outer iter then returns. The outer post-iter dispatch (in
// runNested) sees the same state and emits the real Return op with the
// stored temp values.
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

// runNested returns v*10 + w when v == 1 && w == 1. Both ranges are over
// the same iterator. Without Phase 3a propagation, the inner return would
// only stop the inner iter and the outer body would continue on v == 2.
func runNested() int {
	for v := range count3 {
		for w := range count3 {
			if v == 1 && w == 1 {
				return v*10 + w
			}
		}
	}
	return -1
}

func main() {
	common.Begin("rangefunc_nested_return")

	got := runNested()
	common.AssertEq("got", got, 11)

	common.Done()
	common.Halt()
}
