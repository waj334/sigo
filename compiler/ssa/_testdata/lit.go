// RUN: FileCheck %s

package main

var (
	arrLit    = [4]int{0, 1, 2, 3}
	mapLit    = map[int]int{0: 1, 1: 2, 2: 3}
	sliceLit  = []int{0, 1, 2, 3}
	structLit = struct {
		a, b, c int
	}{a: 0, b: 1, c: 2}
	funcLit = func(a int) int {
		return a
	}

	iArrLit    = [4]any{0, 1, 2, 3}
	iMapLit    = map[int]any{0: 1, 1: 2, 2: 3}
	iSliceLit  = []any{0, 1, 2, 3}
	iStructLit = struct {
		a, b, c any
	}{a: 0, b: 1, c: 2}
)

func testAnonymousFn(f func()) int {
	var a int
	f = func() {
		func() {
			a = 0
		}()
	}
	return a
}

func testIArrayLit() [4]any {
	var a, b, c, d interface{ test() }
	return [4]any{a, b, c, d}
}

func testIMapLit() map[any]any {
	var a, b, c, d interface{ test() }
	return map[any]any{a: a, b: b, c: c, d: d}
}

func testISliceLit() []any {
	var a, b, c, d interface{ test() }
	return []any{a, b, c, d}
}

func testIStructLit() struct {
	a any
	b any
	c any
	d any
} {
	var a, b, c, d interface{ test() }
	return struct {
		a any
		b any
		c any
		d any
	}{a: a, b: b, c: c, d: d}
}
