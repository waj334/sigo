// Test: return inside a range-over-func body propagates to the enclosing
// function and forwards the result values.
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

// runReturn returns early when v == 2. The closure stores (v, v*100) into
// captured temp slots, sets state to the return-sentinel, returns false from
// yield. The iterator observes false and returns. After the iter call, the
// post-iter dispatch sees state == sentinel and emits the actual outer
// Return op with the captured temp values.
func runReturn() (int, int) {
	for v := range count {
		if v == 2 {
			return v, v * 100
		}
	}
	return -1, -1
}

func main() {
	common.Begin("rangefunc_return")

	a, b := runReturn()
	common.AssertEq("a", a, 2)
	common.AssertEq("b", b, 200)

	common.Done()
	common.Halt()
}
