// Test: iter.Pull on a Seq. Exercises the full coroutine chain end-to-end:
// chip-support coro.S asm primitive, chip initCoro, sigo runtime newcoro/
// coroswitch/coroEntry, and the staged stdlib iter.Pull built on top.
package main

import (
	"iter"

	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

// counts yields 10, 20, 30 and then completes.
//
//sigo:stacksize 1234
func counts(yield func(int) bool) {
	for i := 1; i <= 3; i++ {
		if !yield(i * 10) {
			return
		}
	}
}

func main() {
	common.Begin("iter_pull")

	next, stop := iter.Pull(counts)
	defer stop()

	v1, ok1 := next()
	common.AssertEq("v1", v1, 10)
	if !ok1 {
		common.Fail("ok1", "expected ok")
	} else {
		common.Pass("ok1")
	}

	v2, ok2 := next()
	common.AssertEq("v2", v2, 20)
	if !ok2 {
		common.Fail("ok2", "expected ok")
	} else {
		common.Pass("ok2")
	}

	v3, ok3 := next()
	common.AssertEq("v3", v3, 30)
	if !ok3 {
		common.Fail("ok3", "expected ok")
	} else {
		common.Pass("ok3")
	}

	// Past the sequence: next should return zero value and ok=false.
	v4, ok4 := next()
	common.AssertEq("v4", v4, 0)
	if ok4 {
		common.Fail("ok4", "expected !ok")
	} else {
		common.Pass("ok4")
	}

	common.Done()
	common.Halt()
}
