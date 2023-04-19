//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	_ "runtime/arm/cortexm/sam/atsame51g19a"
	_ "runtime/gc/markandsweep"
)

func main() {
	var a any = 0
	var b any = 0

	if a == 0 {
		println(a, b)
	}
}
