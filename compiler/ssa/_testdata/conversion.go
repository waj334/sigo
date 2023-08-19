// RUN: FileCheck %s

package main

import "unsafe"

func signedIntExtend(v int8) int32 {
	return int32(v)
}

func signedIntTrunc(v int32) int8 {
	return int8(v)
}

func unsignedIntExtend(v uint8) uint32 {
	return uint32(v)
}

func unsignedIntTrunc(v uint32) uint8 {
	return uint8(v)
}

func sameInt(v int) int {
	return int(v)
}

func signedIntToFloat(v int) float64 {
	return float64(v)
}

func unsignedIntToFloat(v uint) float64 {
	return float64(v)
}

func runeToString(v rune) string {
	return string(v)
}

func uintptrToUnsafePointer(v uintptr) unsafe.Pointer {
	return unsafe.Pointer(v)
}

func floatToSignedInt(v float64) int {
	return int(v)
}

func floatToUnsignedInt(v float64) uint {
	return uint(v)
}

func floatExtend(v float32) float64 {
	return float64(v)
}

func floatTruncate(v float64) float32 {
	return float32(v)
}

func sameFloat(v float32) float32 {
	return float32(v)
}

func stringToSlice(v string) []byte {
	return []byte(v)
}

func sliceToString(v []byte) string {
	return string(v)
}

func pointerToUnsafePointer(v *int) unsafe.Pointer {
	return unsafe.Pointer(v)
}

func unsafePointerToUintptr(v unsafe.Pointer) uintptr {
	return uintptr(v)
}

func unsafePointerToPointer(v unsafe.Pointer) *int {
	return (*int)(v)
}

func sameUnderlyingType() string {
	type customString string
	var v customString
	return string(v)
}

type uint8Alias uint8

func convertToAlias(v uint8) uint8Alias {
	return uint8Alias(v)
}

func convertLiteralToAlias() uint8Alias {
	return uint8Alias(69)
}

func convertExprToAlias(v uint8) uint8Alias {
	return uint8Alias(v&69) >> 4
}

func unsignedToSigned(v uint64) int64 {
	return int64(v)
}

func aliasToInt(v uint8Alias) uint8 {
	return uint8(v)
}
