//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	_ "runtime/arm/cortexm/sam/atsame51g19a"
	_ "runtime/gc/markandcompact"
)

func main() {
	arr := [10]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	s0 := arr[:]
	s1 := arr[5:]
	s2 := arr[:5]
	s3 := arr[3:6]
	s4 := append(s0, s1...)
	s5 := append(s4, 10, 11, 12, 13, 14, 15)
	i := s3[0]
	s6 := make([]int, i, i)
	println(arr, s0, s1, s2, s3, s4, s5, &s6)
}
