// RUN: FileCheck %s

package main

func testNew() (*int, *int) {
	var ptr = new(int)
	return ptr, new(int)
}

func testEscape() *int {
	var i int
	return &i
}
