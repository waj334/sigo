//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	_ "runtime/arm/cortexm/sam/atsame51g19a"
)

func testFn(byte) {
	// TODO
}

func main() {
	i := 0
	for {
		arr := make([]byte, 1024)
		//arr := make([]byte, (i%8)+1)
		i++
		testFn(arr[0])
	}
}
