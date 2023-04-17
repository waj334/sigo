//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple thumb-none-eabi
package main

import (
	"fmt"
	_ "runtime/gc/markandsweep"
)

type test struct {
	field0 int
	field1 string
	field2 any
}

func testFn(i int) {

}

func main() {
	i := 0
	j := uint(0)
	k := uint64(0)
	testFn(i)
	println(i, j)
	fmt.Println(i, j, k)
}
