// RUN: FileCheck %s

package main

func testAppend(s []int) []int {
	s = append(s, 0, 1, 2, 3)
	s = append(s[1:], s...)
	return s
}

func testCap(arr [4]int, s []int, c chan int) (int, int, int) {
	return cap(arr), cap(s), cap(c)
}

func testClear(m map[int]int, s []int) {
	clear(m)
	clear(s)
}

func testClose(c chan int) {
	close(c)
}

func testComplex() (complex64, complex128) {
	return complex(float32(0), float32(1)), complex(float64(2), float64(3))
}

func testCopySlice(s []int, dest []int) int {
	return copy(dest, s)
}

func testCopyString(s []byte, str string) int {
	return copy(s, str)
}

func testDelete(m map[int]int) {
	delete(m, 0)
}

func testImag(c64 complex64, c128 complex128) (float32, float64) {
	return imag(c64), imag(c128)
}

func testLenString(str string) int {
	return len(str)
}

func testLenArray(arr [4]int) int {
	return len(arr)
}

func testLenSlice(s []int) int {
	return len(s)
}

func testLenMap(m map[int]int) int {
	return len(m)
}

func testLenChan(c chan int) int {
	return len(c)
}

func testMakeChan() (chan int, chan int) {
	c0 := make(chan int)
	c1 := make(chan int, 10)
	return c0, c1
}

func testMakeMap() (map[int]int, map[int]int) {
	m0 := make(map[int]int)
	m1 := make(map[int]int, 10)
	return m0, m1
}

func testMakeSlice() ([]int, []int) {
	s0 := make([]int, 10)
	s1 := make([]int, 0, 10)
	return s0, s1
}

func testPanic(v int) {
	panic(v)
	panic(0)
}

func testPrint() {
	print("this", "is", "test #", 1)
}

func testPrintln() {
	println("this", "is", "test #", 2)
}

func testReal(c64 complex64, c128 complex128) (float32, float64) {
	return real(c64), real(c128)
}

func testRecover() any {
	return recover()
}
