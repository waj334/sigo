// RUN: FileCheck %s

package main

func selectStatement() {
	c := make(chan int)
	select {
	case v := <-c:
		println(v)
	default:
		println("default")
	}
}
