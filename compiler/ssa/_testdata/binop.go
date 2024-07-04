// RUN: FileCheck %s

package main

// CHECK-LABEL: func.func private @main.si(%arg0: !go.i, %arg1: !go.i) -> !go.i {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<!go.i>) -> !go.i
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<!go.i>) -> !go.i
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.divsi %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.remsi %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : !go.i
// CHECK-DAG: %[[#]] = go.shrsi %[[A]], %[[B]] : (!go.i, !go.i) -> !go.i
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (!go.i, !go.i) -> !go.i
func si(a, b int) (result int) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.si16(%arg0: si16, %arg1: si16) -> si16 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<si16>) -> si16
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<si16>) -> si16
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.divsi %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.remsi %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : si16
// CHECK-DAG: %[[#]] = go.shrsi %[[A]], %[[B]] : (si16, si16) -> si16
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (si16, si16) -> si16
func si16(a, b int16) (result int16) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.si32(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<si32>) -> si32
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<si32>) -> si32
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.divsi %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.remsi %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : si32
// CHECK-DAG: %[[#]] = go.shrsi %[[A]], %[[B]] : (si32, si32) -> si32
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (si32, si32) -> si32
func si32(a, b int32) (result int32) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.si64(%arg0: si64, %arg1: si64) -> si64 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<si64>) -> si64
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<si64>) -> si64
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.divsi %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.remsi %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : si64
// CHECK-DAG: %[[#]] = go.shrsi %[[A]], %[[B]] : (si64, si64) -> si64
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (si64, si64) -> si64
func si64(a, b int64) (result int64) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.si8(%arg0: si8, %arg1: si8) -> si8 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<si8>) -> si8
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<si8>) -> si8
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.divsi %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.remsi %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : si8
// CHECK-DAG: %[[#]] = go.shrsi %[[A]], %[[B]] : (si8, si8) -> si8
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (si8, si8) -> si8
func si8(a, b int8) (result int8) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.ui(%arg0: !go.ui, %arg1: !go.ui) -> !go.ui {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<!go.ui>) -> !go.ui
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<!go.ui>) -> !go.ui
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.divui %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.remui %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : !go.ui
// CHECK-DAG: %[[#]] = go.shrui %[[A]], %[[B]] : (!go.ui, !go.ui) -> !go.ui
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (!go.ui, !go.ui) -> !go.ui
func ui(a, b uint) (result uint) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.ui16(%arg0: ui16, %arg1: ui16) -> ui16 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui16>) -> ui16
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui16>) -> ui16
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.divui %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.remui %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : ui16
// CHECK-DAG: %[[#]] = go.shrui %[[A]], %[[B]] : (ui16, ui16) -> ui16
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (ui16, ui16) -> ui16
func ui16(a, b uint16) (result uint16) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.ui32(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui32>) -> ui32
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui32>) -> ui32
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.divui %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.remui %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : ui32
// CHECK-DAG: %[[#]] = go.shrui %[[A]], %[[B]] : (ui32, ui32) -> ui32
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (ui32, ui32) -> ui32
func ui32(a, b uint32) (result uint32) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.ui64(%arg0: ui64, %arg1: ui64) -> ui64 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui64>) -> ui64
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui64>) -> ui64
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.divui %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.remui %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : ui64
// CHECK-DAG: %[[#]] = go.shrui %[[A]], %[[B]] : (ui64, ui64) -> ui64
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (ui64, ui64) -> ui64
func ui64(a, b uint64) (result uint64) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}

// CHECK-LABEL: func.func private @main.ui8(%arg0: ui8, %arg1: ui8) -> ui8 {
// CHECK-DAG: %[[A:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui8>) -> ui8
// CHECK-DAG: %[[B:[0-9]+]] = go.load %[[#]] : (!go.ptr<ui8>) -> ui8
// CHECK-DAG: %[[#]] = go.addi %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.subi %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.muli %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.divui %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.remui %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.and %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.or %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.xor %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.andnot %[[A]], %[[B]] : ui8
// CHECK-DAG: %[[#]] = go.shrui %[[A]], %[[B]] : (ui8, ui8) -> ui8
// CHECK-DAG: %[[#]] = go.shl %[[A]], %[[B]] : (ui8, ui8) -> ui8
func ui8(a, b uint8) (result uint8) {
	result = a + b
	result = a - b
	result = a * b
	result = a / b
	result = a % b

	result = a & b
	result = a | b
	result = a ^ b
	result = a &^ b

	result = a >> b
	result = a << b
	return
}
