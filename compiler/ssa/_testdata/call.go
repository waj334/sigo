// RUN: FileCheck %s

package main

import (
	"omibyte.io/sigo/compiler/ssa/_testdata/src/somepkg"
	"unsafe"
)

//import "omibyte.io/sigo/compiler/ssa/_testdata/src/somepkg"

type someinterface interface {
	method()
}

type someinterface2 interface {
	someinterface
	method2()
}

type sometype struct {
	sometype2
}
type sometype2 struct{}

type sometype3 struct {
	member sometype4
}

type sometype4 struct{}

var (
	g  sometype
	g2 = (*sometype3)(unsafe.Pointer(uintptr(0xDEADBEEF)))
)

type sometype5 int

const (
	constValue sometype5 = 0
)

func (s sometype) method()   {}
func (s *sometype) method2() {}

func (s sometype2) method3()  {}
func (s *sometype2) method4() {}

func (s *sometype4) method() {}

func (s sometype5) method()     {}
func (s *sometype5) ptrMethod() {}

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
	s.method3()
	s.method4()
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

	somepkg.SomePtrValue.Member.Method()
	somepkg.SomePtrValue.Member.PtrMethod()

	somepkg.SomePtrValue.MemberPtr.Method()
	somepkg.SomePtrValue.MemberPtr.PtrMethod()
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

func multipleReturn() (int, uint, bool)

func callMultipleReturn(a int, b uint, c bool) {
	a, b, c = multipleReturn()
}

func globalMethodCall() {
	g.method()
	g.method3()
}

func globalPtrMethodCall() {
	g.method2()
	g.method4()
	g2.member.method()
}

func constValueMethodCall() {
	constValue.method()
}

func packageConstAliasMethodCall() {
	somepkg.SomeConstAliasTypeValue.Method()
}
