// RUN: FileCheck %s

package main

import "unsafe"

const (
	constBoolDecl           bool       = true
	constUntypedBoolDecl               = false
	constComplex64Decl      complex64  = 0.7i
	constComplex128Decl     complex128 = 0.8i
	constUntypedComplexDecl            = 0.9i
	constFloat32Decl        float32    = 0.9
	constFloat64Decl        float64    = 0.95
	constFloatDecl                     = constFloat64Decl
	constUntypedFloatDecl              = 1.1
	constIntDecl            int        = 0
	constInt8Decl           int8       = 1
	constInt16Decl          int16      = 2
	constInt32Decl          int32      = 3
	constInt64Decl          int64      = 4
	constUintDecl           uint       = 5
	constUint8Decl          uint8      = 6
	constUint16Decl         uint16     = 7
	constUint32Decl         uint32     = 8
	constUint64Decl         uint64     = 9
	constUintptrDecl        uintptr    = 10
	constUntypedIntDecl                = 11
	constStringDecl         string     = "const_str"
	constUntypedStringDecl             = "const_untyped_str"
	constExpressionDecl                = float64(1) + (1 << 2)
)

const (
	iota0 = iota
	iota1
	iota2
	iota3
)

const (
	iota_3_0 = iota + 3
	iota_4_1
	iota_5_2
	iota_6_3
)

func constBool(v bool) {
	v = true
}

func constComplex64(v complex64) {
	v = 1 - 0.707i
}

func constComplex128(v complex128) {
	v = 1 - 0.707i
}

func constFloat32(v float32) {
	v = 0.5
}

func constFloat64(v float64) {
	v = 0.5
}

func constInt(v int) {
	v = -1
}

func constUint(v uint) {
	v = 1
}

func constString(v string) {
	v = "conststring"
}

func constUnsafePointer(v unsafe.Pointer) {
	v = unsafe.Pointer(uintptr(0xDEADBEEF))
}

func useItoa(v int) {
	v = iota2
	v = iota_5_2
}

func constExpression() float64 {
	return constExpressionDecl
}
