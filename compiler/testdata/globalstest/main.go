//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	_ "runtime/arm/cortexm/sam/atsame51g19a"
	_ "runtime/gc/markandcompact"
)

type (
	selfref struct {
		i    int
		next *selfref
	}
)

var (
	g0 int
	g1 *selfref
)

func main() {
	g0 = 1
	g0 = 2

	i := 69
	g0 = i

	for i := 0; i < 5; i++ {
		v0 := &selfref{i: i}
		v0.next = g1
		g1 = v0
	}

	println(g0)
}
