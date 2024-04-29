// RUN: FileCheck %s

package main

import "somepkg"

type someinterface interface {
	method()
}

type someinterface2 interface {
	someinterface
	method2()
}

type sometype struct{}

func (s sometype) method()   {}
func (s *sometype) method2() {}

func standardFunc(arg0, arg1 int) (int, int)
func variadicFunc(args ...int)

func callClosure(v int) {
	closure := func(i int) int {
		return v + i
	}
	closure(v)
}

func indirectCall(v int, F func(int)) {
	F(v)
	func() {
		i := 1
		F(v + i)

		func() {
			F(v + i)
		}()
	}()
}

func packageCall() {
	somepkg.SomeFunc()
}

func variadicCall() {
	variadicFunc(0, 1, 2, 3)
}

func call_builtin_make() {
	_ = make(chan int)
	_ = make(chan int, 10)
	_ = make(map[int]int)
	_ = make(map[int]int, 10)
	_ = make([]int, 10)
	_ = make([]int, 10, 20)
}

func funcCall() {
	standardFunc(0, 1)
}

func methodCall(s sometype) {
	s.method()
	s.method2()
}

func interfaceCall(i someinterface) {
	i.method()
}

func goroutineCall(s sometype, i someinterface) {
	var a int
	f := func(b int) int {
		return a + b
	}
	go f(1)
	go standardFunc(0, 1)
	go s.method()
	go i.method()
	go func() {}()
}

func deferCall(s sometype, i someinterface) {
	var a int
	f := func(b int) int {
		return a + b
	}
	defer f(1)
	defer standardFunc(0, 1)
	defer s.method()
	defer i.method()
	defer func() {}()
}

func globalCall() {
	somepkg.SomeValue.Method()
	somepkg.SomeValue.PtrMethod()
	somepkg.SomePtrValue.Method()
	somepkg.SomePtrValue.PtrMethod()
}

func elementRecv() {
	var arr [4]somepkg.SomeType
	arr[0].Method()
}

func elementRecv2() {
	var arr [4]*somepkg.SomeType
	arr[0].Method()
}

func somefunc(s someinterface)
func argImplementsInterface(s someinterface2) {
	somefunc(s)
}

func somefunc2(a any)
func argIsAssignable() {
	somefunc2(0)
}

type funcAlias func()

func callFuncAlias(f funcAlias) {
	f()
}
