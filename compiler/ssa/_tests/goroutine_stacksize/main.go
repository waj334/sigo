package main

import (
	"pkg.si-go.dev/sigo/compiler/ssa/_tests/common"
)

func init() {
	common.Setup()
}

func main() {
	common.Begin("goroutine_stacksize")

	//sigo:stacksize 512
	go func() {}()
	common.Pass("ok1")

	//sigo:stacksizee 256
	go fn()
	common.Pass("ok2")

	go fn2()
	common.Pass("ok3")

	common.Done()
	common.Halt()
}

//sigo:stacksize 128
func fn() {}

//sigo:stacksize 64
func fn2() {}
