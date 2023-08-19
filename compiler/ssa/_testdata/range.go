// RUN: FileCheck %s

package main

func rangeArray(arr [4]int, i int) {
	for i = range arr {
	}
}

func rangeArrayLiteral(i int) {
	for i = range [4]int{0, 1, 2, 3} {
	}
}

func rangeArrayKeyValue(arr [4]int, i, v int) {
	for i, v = range arr {
	}
}

func rangeSlice(s []int, i int) {
	for i = range s {
	}
}

func rangeSliceKeyValue(s []int, i, v int) {
	for i, v = range s {
	}
}

func rangeSliceLiteral(i int) {
	for i = range []int{0, 1, 2, 3} {
	}
}

func rangeString(s string, i int) {
	for i = range s {
	}
}

func rangeStringLiteral(i int) {
	for i = range "thisisastring" {
	}
}

func rangeStringKeyValue(s string, i int, v rune) {
	for i, v = range s {
	}
}

func rangeMap(m map[int]int, k int) {
	for k = range m {
	}
}

func rangeMapKeyValue(m map[int]int, k, v int) {
	for k, v = range m {
	}
}

func rangeMapLiteral(k int) {
	for k = range map[int]int{0: 1, 1: 2, 2: 3} {
	}
}

func rangeChan(c chan int, v int) {
	for v = range c {
	}
}
