//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	"omibyte.io/sigo/cmd/sigoc/testdata/pgk0/testpkg"

	_ "runtime/arm/cortexm/sam/atsame51g19a"
	_ "runtime/gc/markandsweep"

	"fmt"
)

type test struct {
	field0 int
	field1 string
	field2 any
}

func testFn(i int) {
	arr := [4]int{0, 1, 2, 3}
	slice := []int{5, 6, 7, 8}

	for j := 0; j < len(arr); j++ {
		a := arr[j]
		b := slice[j]
		k := (a + b) * i
		println(k)
	}
}

func main() {
	var i0 any = 0
	var i1 any = "test"
	var i2 any = []byte{0, 1, 2, 3, 4}
	var i3 any = false
	var i4 any = float32(0)
	var i5 any = float64(0)

	var a any = 0
	var b any = 0

	if a == b {
		println(a, b)
	}

	println(i0, i1, i2, i3, i4, i5)

	testpkg.TestFunc()
	i := 0
	j := uint(0)
	k := uint64(0)
	testFn(i)
	println(i, j)
	fmt.Println(i, j, k)
}
