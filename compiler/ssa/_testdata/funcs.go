// RUN: FileCheck %s

package main

func forwardDeclare() (a, b int)

func empty() {}

func emptyWithReturn() { return }

func singleReturn(i int) int { return i }

func multipleReturns(a int, b int) (int, int) { return a, b }

func multipleWithUnnamedParam(_ int, a int, b int) (int, int) { return a, b }

func variadic(a int, b ...int) (int, []int) { return a, b }

func namedResults(a, b int) (c, d int) { c, d = a, b; return }

func namedResults2(a, b int) (_, d int) { _, d = a, b; return }

func nameResults3(a, b int) (c, d int) { return forwardDeclare() }

func Exported() {}

type T0 struct{}

func (t *T0) typeMethodPointerReceiver() {}

func (t T0) typeMethodWithMultipleReturns(a, b int) (int, int) { return a, b }

func (t T0) Exported() {}

func returnNils() (*int, *float32, *bool) {
	return nil, nil, nil
}

func returnNils2() (a, b, c *int) {
	return nil, nil, nil
}

func returnBasicLits() (int, bool, float32) {
	return 0, true, 0.5
}
