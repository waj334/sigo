// RUN: FileCheck %s

package main

func genericFn[P any](p P) P {
	return p
}

func testGenericFn() {
	genericFn(0)
	genericFn(0.0)
}

type genericStruct[P any] struct {
	p P
	i interface {
		generateIfaceMethod(P)
	}
}

func (g genericStruct[P]) genericMember() genericStruct[P] {
	var i interface {
		generateIfaceMethod(P)
	}
	g.i = i
	return g
}

func testGenericStruct() (genericStruct[int], genericStruct[float64]) {
	s0 := genericStruct[int]{p: 0}
	s0.genericMember()
	s1 := genericStruct[float64]{p: 0}
	s1.genericMember()
	return s0, s1
}
