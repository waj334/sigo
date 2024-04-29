// RUN: FileCheck %s

package main

import "unsafe"

func inferUntypedType(val uintptr) uintptr {
	const constInt = 1
	return (val + constInt) - (val % constInt)
}

func inferUntypedType2() float64 {
	r := int64(1)
	return float64(r) + (1 << 2)
}

func inferUntypedType3() bool {
	var i int64
	return i > 1<<31-1-1
}

func inferUntypedType4(v uint8) uint8 {
	v |= 0x1 << 0
	return v
}

func inferUntypedType5() int16 {
	// Test negation unary op
	return -1
}

func compareNil(x *int) bool {
	return x == nil
}

func compareNil2(x any) bool {
	return x == nil
}

func fn()
func compareFunc() bool {
	return fn != nil
}

func compareFunc2(f func()) bool {
	return f != nil
}

type opsS struct {
	r unsafe.Pointer
}

type opsS2 struct {
	r *opsS
}

func assignAddressToStructMember(s *opsS2) {
	s.r = &opsS{
		r: nil,
	}
}
