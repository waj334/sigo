// Test: break inside a range-over-func body.
//
// Verifies that `break` inside the body causes the yield closure to return
// false, the iterator observes false, stops, and control falls through to
// after the range statement.
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

// count yields 0..9 and respects yield's return value.
func count(yield func(int) bool) {
	for i := 0; i < 10; i++ {
		if !yield(i) {
			return
		}
	}
}

func main() {
	common.Begin("rangefunc_break")

	calls := 0
	last := -1
	for v := range count {
		calls++
		last = v
		if v == 3 {
			break
		}
	}

	common.AssertEq("calls", calls, 4) // yields 0,1,2,3 then break
	common.AssertEq("last", last, 3)

	common.Done()
	common.Halt()
}
