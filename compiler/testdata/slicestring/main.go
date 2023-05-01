//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	_ "runtime/arm/cortexm/sam/atsame51g19a"
	_ "runtime/gc/markandsweep"
)

func main() {
	arr := "0123456789"
	s0 := arr[:]
	s1 := arr[5:]
	s2 := arr[:5]
	s3 := arr[3:6]
	s4 := arr + "876543210"
	len4 := len(s4)

	var b []byte
	b = append(b, "bar"...)
	n := copy(b, "foo")
	println(arr, s0, s1, s2, s3, s4, len4, b, n)
}
