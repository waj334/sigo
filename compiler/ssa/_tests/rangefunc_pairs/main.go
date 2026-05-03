// Test: range-over-func with a two-value Seq2.
//
// Iterates a `func(yield func(int, int) bool)` and verifies both key and
// value reach the loop body.
package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func pairs(yield func(int, int) bool) {
	for i := 0; i < 4; i++ {
		if !yield(i, i*10) {
			return
		}
	}
}

func main() {
	common.Begin("rangefunc_pairs")

	keySum := 0
	valSum := 0
	calls := 0
	for k, v := range pairs {
		keySum += k
		valSum += v
		calls++
	}

	common.AssertEq("calls", calls, 4)
	common.AssertEq("keySum", keySum, 0+1+2+3)
	common.AssertEq("valSum", valSum, 0+10+20+30)

	common.Done()
	common.Halt()
}
