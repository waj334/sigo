// RUN: FileCheck %s

package main

import "somepkg"

type embeddedStruct struct {
	e int
	f int
	g int
}

type embeddedStructPtr struct {
	h int
	i int
	j int
}

type memberStruct struct {
	k int
	l int
	m int
}

type selectorStruct struct {
	a int
	b int
	c int
	d func()
	embeddedStruct
	*embeddedStructPtr
	o memberStruct
	p *memberStruct
}

var g selectorStruct
var gptr *selectorStruct

func (s selectorStruct) fn() {}

func selectStruct(s selectorStruct) int {
	return s.b
}

func selectPtrToStruct(s *selectorStruct) int {
	return s.c
}

func selectMemberStruct(s selectorStruct) int {
	return s.o.m
}

func selectMemberStructPtr(s selectorStruct) int {
	return s.p.m
}

func selectEmbeddedPtrToStruct(s *selectorStruct) int {
	return s.f
}

func selectPtrEmbeddedPtrToStruct(s *selectorStruct) int {
	return s.i
}

func selectMethod(s selectorStruct) func() {
	return s.fn
}

func selectPtrMethod(s *selectorStruct) func() {
	return s.fn
}

func selectMethodMember(s selectorStruct) func() {
	return s.d
}

func selectPtrMethodMember(s *selectorStruct) func() {
	return s.d
}

func selectGlobalMember() int {
	return g.a
}

func selectGlobalPtrMember() int {
	return gptr.a
}

func selectPackageGlobalMember() int {
	return somepkg.SomeValue.A
}

func selectPackageGlobalPtrMember() int {
	return somepkg.SomePtrValue.A
}
