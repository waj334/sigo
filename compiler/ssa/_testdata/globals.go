// RUN: FileCheck %s

package main

import "unsafe"

type SOME_TYPE struct{}

var (
	globalBoolDecl             bool           = true
	globalUntypedBoolDecl                     = false
	globalComplex64Decl        complex64      = 0.7i
	globalComplex128Decl       complex128     = 0.8i
	globalUntypedComplexDecl                  = 0.9i
	globalFloat32Decl          float32        = 0.9
	globalFloat64Decl          float64        = 0.95
	globalFloatDecl                           = globalFloat64Decl + 1
	globalUntypedFloatDecl                    = 1.1
	globalIntDecl              int            = 0
	globalInt8Decl             int8           = 1
	globalInt16Decl            int16          = 2
	globalInt32Decl            int32          = 3
	globalInt64Decl            int64          = 4
	globalUintDecl             uint           = 5
	globalUint8Decl            uint8          = 6
	globalUint16Decl           uint16         = 7
	globalUint32Decl           uint32         = 8
	globalUint64Decl           uint64         = 9
	globalUintptrDecl          uintptr        = 10
	globalUntypedIntDecl                      = 11
	globalStringDecl           string         = "const_str"
	globalUntypedStringDecl                   = "const_untyped_str"
	globalunsafePointerDecl    unsafe.Pointer = unsafe.Pointer(uintptr(0x41006000))
	globalPointerDecl                         = (*SOME_TYPE)(unsafe.Pointer(uintptr(0x41006000)))
	globalNilunsafePointerDecl unsafe.Pointer = nil
	globalStruct                              = struct {
		A int
		B float32
		C bool
	}{
		A: 0,
		B: 1,
		C: true,
	}
	globalInterface    any = SOME_TYPE{}
	globalInterfacePtr any = &SOME_TYPE{}
)

func initializer() int

var globalIntializeWithFunc = initializer() + 1
