//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	mcu "runtime/arm/cortexm/sam/atsame51g19a"
)

/*
func testFn(byte) {
	// TODO
}

func testFn3(b []byte) {
	println(b)
}*/

func main() {
	mcu.GCLK.GENCTRL[1].SetGENEN(true)
	/*c := uint8(52)
	testFn2 := func(b byte) byte {
		return b + c
	}

	i := 0
	for {
		arr := make([]byte, (i%8)+1)
		i++
		testFn(arr[0])
		testFn2(arr[0])
		testFn3(arr)
	}*/
}
