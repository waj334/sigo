package compiler

import (
	"context"
	"omibyte.io/sigo/llvm"
	"reflect"
	"strings"
	"testing"

	"omibyte.io/sigo/compiler/testutil"
)

var target *Target

func init() {
	// Set up compile target
	llvm.InitializeAllTargets()
	llvm.InitializeAllTargetInfos()
	llvm.InitializeAllTargetMCs()
	llvm.InitializeAllAsmParsers()
	llvm.InitializeAllAsmPrinters()

	target, _ = NewTargetFromMap(map[string]string{
		"architecture": "arm",
		"cpu":          "cortex-m4",
		"triple":       "arm-none-eabi",
	})

	target.Initialize()
}

func TestBinop(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		expected string
	}{
		/***************************************************/
		/************** ADD ********************************/
		/***************************************************/
		{
			"addInt8",
			`func wrapper(x, y int8) int8 { return x + y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = add i8 %2, %3
  ret i8 %4`,
		},
		{
			"addUint8",
			`func wrapper(x, y uint8) uint8 { return x + y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = add i8 %2, %3
  ret i8 %4`,
		},
		{
			"addInt16",
			`func wrapper(x, y int16) int16 { return x + y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = add i16 %2, %3
  ret i16 %4`,
		},
		{
			"addUint16",
			`func wrapper(x, y uint16) uint16 { return x + y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = add i16 %2, %3
  ret i16 %4`,
		},
		{
			"addInt",
			`func wrapper(x, y int) int { return x + y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = add i32 %2, %3
  ret i32 %4`,
		},
		{
			"addUint",
			`func wrapper(x, y uint) uint { return x + y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = add i32 %2, %3
  ret i32 %4`,
		},
		{
			"addFloat32",
			`func wrapper(x, y float32) float32 { return x + y }`,
			`
wrapper.entry:
  %x = alloca float, align 4
  store float 0.000000e+00, ptr %x, align 4
  store float %0, ptr %x, align 4
  %y = alloca float, align 4
  store float 0.000000e+00, ptr %y, align 4
  store float %1, ptr %y, align 4
  %2 = load float, ptr %x, align 4
  %3 = load float, ptr %y, align 4
  %4 = fadd float %2, %3
  ret float %4`,
		},
		{
			"addFloat64",
			`func wrapper(x, y float64) float64 { return x + y }`,
			`
wrapper.entry:
  %x = alloca double, align 8
  store double 0.000000e+00, ptr %x, align 8
  store double %0, ptr %x, align 8
  %y = alloca double, align 8
  store double 0.000000e+00, ptr %y, align 8
  store double %1, ptr %y, align 8
  %2 = load double, ptr %x, align 8
  %3 = load double, ptr %y, align 8
  %4 = fadd double %2, %3
  ret double %4`,
		},
		/***************************************************/
		/************** SUB ********************************/
		/***************************************************/
		{
			"subInt8",
			`func wrapper(x, y int8) int8 { return x - y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = sub i8 %2, %3
  ret i8 %4`,
		},
		{
			"subUint8",
			`func wrapper(x, y uint8) uint8 { return x - y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = sub i8 %2, %3
  ret i8 %4`,
		},
		{
			"subInt16",
			`func wrapper(x, y int16) int16 { return x - y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = sub i16 %2, %3
  ret i16 %4`,
		},
		{
			"subUint16",
			`func wrapper(x, y uint16) uint16 { return x - y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = sub i16 %2, %3
  ret i16 %4`,
		},
		{
			"subInt",
			`func wrapper(x, y int) int { return x - y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = sub i32 %2, %3
  ret i32 %4`,
		},
		{
			"subUint",
			`func wrapper(x, y uint) uint { return x - y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = sub i32 %2, %3
  ret i32 %4`,
		},
		{
			"subFloat32",
			`func wrapper(x, y float32) float32 { return x - y }`,
			`
wrapper.entry:
  %x = alloca float, align 4
  store float 0.000000e+00, ptr %x, align 4
  store float %0, ptr %x, align 4
  %y = alloca float, align 4
  store float 0.000000e+00, ptr %y, align 4
  store float %1, ptr %y, align 4
  %2 = load float, ptr %x, align 4
  %3 = load float, ptr %y, align 4
  %4 = fsub float %2, %3
  ret float %4`,
		},
		{
			"subFloat64",
			`func wrapper(x, y float64) float64 { return x - y }`,
			`
wrapper.entry:
  %x = alloca double, align 8
  store double 0.000000e+00, ptr %x, align 8
  store double %0, ptr %x, align 8
  %y = alloca double, align 8
  store double 0.000000e+00, ptr %y, align 8
  store double %1, ptr %y, align 8
  %2 = load double, ptr %x, align 8
  %3 = load double, ptr %y, align 8
  %4 = fsub double %2, %3
  ret double %4`,
		},
		/***************************************************/
		/************** MUL ********************************/
		/***************************************************/
		{
			"mulInt8",
			`func wrapper(x, y int8) int8 { return x * y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = mul i8 %2, %3
  ret i8 %4`,
		},
		{
			"mulUint8",
			`func wrapper(x, y uint8) uint8 { return x * y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = mul i8 %2, %3
  ret i8 %4`,
		},
		{
			"mulInt16",
			`func wrapper(x, y int16) int16 { return x * y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = mul i16 %2, %3
  ret i16 %4`,
		},
		{
			"mulUint16",
			`func wrapper(x, y uint16) uint16 { return x * y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = mul i16 %2, %3
  ret i16 %4`,
		},
		{
			"mulInt",
			`func wrapper(x, y int) int { return x * y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = mul i32 %2, %3
  ret i32 %4`,
		},
		{
			"mulUint",
			`func wrapper(x, y uint) uint { return x * y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = mul i32 %2, %3
  ret i32 %4`,
		},
		{
			"mulFloat32",
			`func wrapper(x, y float32) float32 { return x * y }`,
			`
wrapper.entry:
  %x = alloca float, align 4
  store float 0.000000e+00, ptr %x, align 4
  store float %0, ptr %x, align 4
  %y = alloca float, align 4
  store float 0.000000e+00, ptr %y, align 4
  store float %1, ptr %y, align 4
  %2 = load float, ptr %x, align 4
  %3 = load float, ptr %y, align 4
  %4 = fmul float %2, %3
  ret float %4`,
		},
		{
			"mulFloat64",
			`func wrapper(x, y float64) float64 { return x * y }`,
			`
wrapper.entry:
  %x = alloca double, align 8
  store double 0.000000e+00, ptr %x, align 8
  store double %0, ptr %x, align 8
  %y = alloca double, align 8
  store double 0.000000e+00, ptr %y, align 8
  store double %1, ptr %y, align 8
  %2 = load double, ptr %x, align 8
  %3 = load double, ptr %y, align 8
  %4 = fmul double %2, %3
  ret double %4`,
		},
		/***************************************************/
		/************** DIV ********************************/
		/***************************************************/
		{
			"divInt8",
			`func wrapper(x, y int8) int8 { return x / y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = sdiv i8 %2, %3
  ret i8 %4`,
		},
		{
			"divUint8",
			`func wrapper(x, y uint8) uint8 { return x / y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = udiv i8 %2, %3
  ret i8 %4`,
		},
		{
			"divInt16",
			`func wrapper(x, y int16) int16 { return x / y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = sdiv i16 %2, %3
  ret i16 %4`,
		},
		{
			"divUint16",
			`func wrapper(x, y uint16) uint16 { return x / y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = udiv i16 %2, %3
  ret i16 %4`,
		},
		{
			"divInt",
			`func wrapper(x, y int) int { return x / y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = sdiv i32 %2, %3
  ret i32 %4`,
		},
		{
			"divUint",
			`func wrapper(x, y uint) uint { return x / y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = udiv i32 %2, %3
  ret i32 %4`,
		},
		{
			"divFloat32",
			`func wrapper(x, y float32) float32 { return x / y }`,
			`
wrapper.entry:
  %x = alloca float, align 4
  store float 0.000000e+00, ptr %x, align 4
  store float %0, ptr %x, align 4
  %y = alloca float, align 4
  store float 0.000000e+00, ptr %y, align 4
  store float %1, ptr %y, align 4
  %2 = load float, ptr %x, align 4
  %3 = load float, ptr %y, align 4
  %4 = fdiv float %2, %3
  ret float %4`,
		},
		{
			"divFloat64",
			`func wrapper(x, y float64) float64 { return x / y }`,
			`
wrapper.entry:
  %x = alloca double, align 8
  store double 0.000000e+00, ptr %x, align 8
  store double %0, ptr %x, align 8
  %y = alloca double, align 8
  store double 0.000000e+00, ptr %y, align 8
  store double %1, ptr %y, align 8
  %2 = load double, ptr %x, align 8
  %3 = load double, ptr %y, align 8
  %4 = fdiv double %2, %3
  ret double %4`,
		},
		/***************************************************/
		/************** rem ********************************/
		/***************************************************/
		{
			"remInt8",
			`func wrapper(x, y int8) int8 { return x % y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = srem i8 %2, %3
  ret i8 %4`,
		},
		{
			"remUint8",
			`func wrapper(x, y uint8) uint8 { return x % y }`,
			`
wrapper.entry:
  %x = alloca i8, align 1
  store i8 0, ptr %x, align 1
  store i8 %0, ptr %x, align 1
  %y = alloca i8, align 1
  store i8 0, ptr %y, align 1
  store i8 %1, ptr %y, align 1
  %2 = load i8, ptr %x, align 1
  %3 = load i8, ptr %y, align 1
  %4 = urem i8 %2, %3
  ret i8 %4`,
		},
		{
			"remInt16",
			`func wrapper(x, y int16) int16 { return x % y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = srem i16 %2, %3
  ret i16 %4`,
		},
		{
			"remUint16",
			`func wrapper(x, y uint16) uint16 { return x % y }`,
			`
wrapper.entry:
  %x = alloca i16, align 2
  store i16 0, ptr %x, align 2
  store i16 %0, ptr %x, align 2
  %y = alloca i16, align 2
  store i16 0, ptr %y, align 2
  store i16 %1, ptr %y, align 2
  %2 = load i16, ptr %x, align 2
  %3 = load i16, ptr %y, align 2
  %4 = urem i16 %2, %3
  ret i16 %4`,
		},
		{
			"remInt",
			`func wrapper(x, y int) int { return x % y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = srem i32 %2, %3
  ret i32 %4`,
		},
		{
			"remUint",
			`func wrapper(x, y uint) uint { return x % y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = urem i32 %2, %3
  ret i32 %4`,
		},
		/***************************************************/
		/***************************************************/
		/***************************************************/
		{
			"AND",
			`func wrapper(x, y bool) bool { return x && y }`,
			`
wrapper.entry:
  %x = alloca i1, align 1
  store i1 false, ptr %x, align 1
  store i1 %0, ptr %x, align 1
  %y = alloca i1, align 1
  store i1 false, ptr %y, align 1
  store i1 %1, ptr %y, align 1
  %2 = load i1, ptr %x, align 1
  br i1 %2, label %wrapper.binop.rhs, label %wrapper.binop.done

wrapper.binop.rhs:                                ; preds = %wrapper.entry
  %3 = load i1, ptr %y, align 1
  br label %wrapper.binop.done

wrapper.binop.done:                               ; preds = %wrapper.binop.rhs, %wrapper.entry
  %4 = phi i1 [ false, %wrapper.entry ], [ %3, %wrapper.binop.rhs ]
  ret i1 %4`,
		},
		{
			"OR",
			`func wrapper(x, y bool) bool { return x || y }`,
			`
wrapper.entry:
  %x = alloca i1, align 1
  store i1 false, ptr %x, align 1
  store i1 %0, ptr %x, align 1
  %y = alloca i1, align 1
  store i1 false, ptr %y, align 1
  store i1 %1, ptr %y, align 1
  %2 = load i1, ptr %x, align 1
  br i1 %2, label %wrapper.binop.done, label %wrapper.binop.rhs

wrapper.binop.rhs:                                ; preds = %wrapper.entry
  %3 = load i1, ptr %y, align 1
  br label %wrapper.binop.done

wrapper.binop.done:                               ; preds = %wrapper.binop.rhs, %wrapper.entry
  %4 = phi i1 [ true, %wrapper.entry ], [ %3, %wrapper.binop.rhs ]
  ret i1 %4`,
		},
		{
			"XOR",
			`func wrapper(x, y int) int { return x ^ y }`,
			`
wrapper.entry:
  %x = alloca i32, align 4
  store i32 0, ptr %x, align 4
  store i32 %0, ptr %x, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %y, align 4
  store i32 %1, ptr %y, align 4
  %2 = load i32, ptr %x, align 4
  %3 = load i32, ptr %y, align 4
  %4 = xor i32 %2, %3
  ret i32 %4`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pkg := testutil.ProgramString(test.src)
			cc, ctx := NewCompiler(Options{
				Target:            target,
				Symbols:           nil,
				GenerateDebugInfo: false,
				Verbosity:         0,
				PathMappings:      nil,
			})

			if err := cc.CompilePackage(context.Background(), ctx, pkg); err != nil {
				t.Fatal(err)
			}

			// Strip debug info
			llvm.StripModuleDebugInfo(cc.module)

			// Get IR
			block := llvm.GetEntryBasicBlock(llvm.GetNamedFunction(cc.module, "main.wrapper"))
			var ir string
			for block != nil {
				ir += llvm.PrintValueToString(block)
				block = llvm.GetNextBasicBlock(block)
			}

			if !reflect.DeepEqual(strings.TrimSpace(ir), strings.TrimSpace(test.expected)) {
				t.Errorf("expected:\n%s\n\ngot:\n%s", test.expected, ir)
			}
		})
	}
}
