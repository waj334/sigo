// RUN: go-opt -allow-unregistered-dialect %s -split-input-file | go-opt -allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @addc
func.func @addc(%arg0 : !go.complex64, %arg1 : !go.complex64) -> !go.complex64 {
    // CHECK: %0 = go.addc %arg0, %arg1 : !go.complex64
    %0 = go.addc %arg0, %arg1 : !go.complex64
    return %0 : !go.complex64
}

// CHECK-LABEL: @addf
func.func @addf(%arg0 : f64, %arg1 : f64) -> f64 {
    // CHECK: %0 = go.addf %arg0, %arg1 : f64
    %0 = go.addf %arg0, %arg1 : f64
    return %0 : f64
}

// CHECK-LABEL: @addi
func.func @addi(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = go.addi %arg0, %arg1 : i64
    %0 = go.addi %arg0, %arg1 : i64
    return %0 : i64
}

// CHECK-LABEL: @addstr
func.func @addstr(%arg0 : !go.string, %arg1 : !go.string) -> !go.string {
    // CHECK: %0 = go.addstr %arg0, %arg1 : !go.string
    %0 = go.addstr %arg0, %arg1 : !go.string
    return %0 : !go.string
}

// CHECK-LABEL: @and
func.func @and(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.and %arg0, %arg1 : i32
    %0 = go.and %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @andnot
func.func @andnot(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.andnot %arg0, %arg1 : i32
    %0 = go.andnot %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @cmpc
func.func @cmpc(%arg0 : !go.complex64, %arg1 : !go.complex64) -> i1 {
    // CHECK: %0 = go.cmpc eq, %arg0, %arg1 : !go.complex64
    %0 = go.cmpc eq, %arg0, %arg1 : !go.complex64

    // CHECK: %1 = go.cmpc gt, %arg0, %arg1 : !go.complex64
    %1 = go.cmpc gt, %arg0, %arg1 : !go.complex64

    // CHECK: %2 = go.cmpc ge, %arg0, %arg1 : !go.complex64
    %2 = go.cmpc ge, %arg0, %arg1 : !go.complex64

    // CHECK: %3 = go.cmpc lt, %arg0, %arg1 : !go.complex64
    %3 = go.cmpc lt, %arg0, %arg1 : !go.complex64

    // CHECK: %4 = go.cmpc le, %arg0, %arg1 : !go.complex64
    %4 = go.cmpc le, %arg0, %arg1 : !go.complex64

    // CHECK: %5 = go.cmpc ne, %arg0, %arg1 : !go.complex64
    %5 = go.cmpc ne, %arg0, %arg1 : !go.complex64

    return %0 : i1
}

// CHECK-LABEL: @cmpf
func.func @cmpf(%arg0 : f32, %arg1 : f32) -> i1 {
    // CHECK: %0 = go.cmpf eq, %arg0, %arg1 : f32
    %0 = go.cmpf eq, %arg0, %arg1 : f32

    // CHECK: %1 = go.cmpf gt, %arg0, %arg1 : f32
    %1 = go.cmpf gt, %arg0, %arg1 : f32

    // CHECK: %2 = go.cmpf ge, %arg0, %arg1 : f32
    %2 = go.cmpf ge, %arg0, %arg1 : f32

    // CHECK: %3 = go.cmpf lt, %arg0, %arg1 : f32
    %3 = go.cmpf lt, %arg0, %arg1 : f32

    // CHECK: %4 = go.cmpf le, %arg0, %arg1 : f32
    %4 = go.cmpf le, %arg0, %arg1 : f32

    // CHECK: %5 = go.cmpf ne, %arg0, %arg1 : f32
    %5 = go.cmpf ne, %arg0, %arg1 : f32

    return %0 : i1
}

// CHECK-LABEL: @cmpi
func.func @cmpi(%arg0 : i32, %arg1 : i32) -> i1 {
    // CHECK: %0 = go.cmpi eq, %arg0, %arg1 : i32
    %0 = go.cmpi eq, %arg0, %arg1 : i32

    // CHECK: %1 = go.cmpi ne, %arg0, %arg1 : i32
    %1 = go.cmpi ne, %arg0, %arg1 : i32

    // CHECK: %2 = go.cmpi slt, %arg0, %arg1 : i32
    %2 = go.cmpi slt, %arg0, %arg1 : i32

    // CHECK: %3 = go.cmpi sle, %arg0, %arg1 : i32
    %3 = go.cmpi sle, %arg0, %arg1 : i32

    // CHECK: %4 = go.cmpi sgt, %arg0, %arg1 : i32
    %4 = go.cmpi sgt, %arg0, %arg1 : i32

    // CHECK: %5 = go.cmpi sge, %arg0, %arg1 : i32
    %5 = go.cmpi sge, %arg0, %arg1 : i32

    // CHECK: %6 = go.cmpi ult, %arg0, %arg1 : i32
    %6 = go.cmpi ult, %arg0, %arg1 : i32

    // CHECK: %7 = go.cmpi ule, %arg0, %arg1 : i32
    %7 = go.cmpi ule, %arg0, %arg1 : i32

    // CHECK: %8 = go.cmpi ugt, %arg0, %arg1 : i32
    %8 = go.cmpi ugt, %arg0, %arg1 : i32

    // CHECK: %9 = go.cmpi uge, %arg0, %arg1 : i32
    %9 = go.cmpi uge, %arg0, %arg1 : i32

    return %0 : i1
}

// CHECK-LABEL: @divc
func.func @divc(%arg0 : !go.complex64, %arg1 : !go.complex64) -> !go.complex64 {
    // CHECK: %0 = go.divc %arg0, %arg1 : !go.complex64
    %0 = go.divc %arg0, %arg1 : !go.complex64
    return %0 : !go.complex64
}

// CHECK-LABEL: @divf
func.func @divf(%arg0 : f64, %arg1 : f64) -> f64 {
    // CHECK: %0 = go.divf %arg0, %arg1 : f64
    %0 = go.divf %arg0, %arg1 : f64
    return %0 : f64
}

// CHECK-LABEL: @divui
func.func @divui(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = go.divui %arg0, %arg1 : i64
    %0 = go.divui %arg0, %arg1 : i64
    return %0 : i64
}

// CHECK-LABEL: @divsi
func.func @divsi(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = go.divsi %arg0, %arg1 : i64
    %0 = go.divsi %arg0, %arg1 : i64
    return %0 : i64
}

// CHECK-LABEL: @mulc
func.func @mulc(%arg0 : !go.complex64, %arg1 : !go.complex64) -> !go.complex64 {
    // CHECK: %0 = go.mulc %arg0, %arg1 : !go.complex64
    %0 = go.mulc %arg0, %arg1 : !go.complex64
    return %0 : !go.complex64
}

// CHECK-LABEL: @mulf
func.func @mulf(%arg0 : f64, %arg1 : f64) -> f64 {
    // CHECK: %0 = go.mulf %arg0, %arg1 : f64
    %0 = go.mulf %arg0, %arg1 : f64
    return %0 : f64
}

// CHECK-LABEL: @muli
func.func @muli(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = go.muli %arg0, %arg1 : i64
    %0 = go.muli %arg0, %arg1 : i64
    return %0 : i64
}

// CHECK-LABEL: @or
func.func @or(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.or %arg0, %arg1 : i32
    %0 = go.or %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @shl
func.func @shl(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.shl %arg0, %arg1 : i32
    %0 = go.shl %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @shrui
func.func @shrui(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.shrui %arg0, %arg1 : i32
    %0 = go.shrui %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @shrsi
func.func @shrsi(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.shrsi %arg0, %arg1 : i32
    %0 = go.shrsi %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @remf
func.func @remf(%arg0 : f32, %arg1 : f32) -> f32 {
    // CHECK: %0 = go.remf %arg0, %arg1 : f32
    %0 = go.remf %arg0, %arg1 : f32
    return %0 : f32
}

// CHECK-LABEL: @remsi
func.func @remsi(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.remsi %arg0, %arg1 : i32
    %0 = go.remsi %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @remui
func.func @remui(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.remui %arg0, %arg1 : i32
    %0 = go.remui %arg0, %arg1 : i32
    return %0 : i32
}

// CHECK-LABEL: @subc
func.func @subc(%arg0 : !go.complex64, %arg1 : !go.complex64) -> !go.complex64 {
    // CHECK: %0 = go.subc %arg0, %arg1 : !go.complex64
    %0 = go.subc %arg0, %arg1 : !go.complex64
    return %0 : !go.complex64
}

// CHECK-LABEL: @subf
func.func @subf(%arg0 : f64, %arg1 : f64) -> f64 {
    // CHECK: %0 = go.subf %arg0, %arg1 : f64
    %0 = go.subf %arg0, %arg1 : f64
    return %0 : f64
}

// CHECK-LABEL: @subi
func.func @subi(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = go.subi %arg0, %arg1 : i64
    %0 = go.subi %arg0, %arg1 : i64
    return %0 : i64
}

// CHECK-LABEL: @xor
func.func @xor(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = go.xor %arg0, %arg1 : i32
    %0 = go.xor %arg0, %arg1 : i32
    return %0 : i32
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @comp
func.func @comp(%arg0 : i32) -> i32 {
    // CHECK: %0 = go.comp %arg0 : i32
    %0 = go.comp %arg0 : i32
    return %0 : i32
}

// CHECK-LABEL: @negc
func.func @negc(%arg0 : !go.complex64) -> !go.complex64 {
    // CHECK: %0 = go.negc %arg0 : !go.complex64
    %0 = go.negc %arg0 : !go.complex64
    return %0 : !go.complex64
}

// CHECK-LABEL: @negf
func.func @negf(%arg0 : f32) -> f32 {
    // CHECK: %0 = go.negf %arg0 : f32
    %0 = go.negf %arg0 : f32
    return %0 : f32
}

// CHECK-LABEL: @negi
func.func @negi(%arg0 : i32) -> i32 {
    // CHECK: %0 = go.negi %arg0 : i32
    %0 = go.negi %arg0 : i32
    return %0 : i32
}

// CHECK-LABEL: @not
func.func @not(%arg0 : i32) -> i32 {
    // CHECK: %0 = go.not %arg0 : i32
    %0 = go.not %arg0 : i32
    return %0 : i32
}

// CHECK-LABEL: @recv
func.func @recv(%arg0 : !go.chan<i32>) -> i32 {
    // CHECK: %0 = go.recv %arg0 : i32
    %0 = go.recv %arg0 : i32
    return %0 : i32
}

//===----------------------------------------------------------------------===//
// Memory Allocation Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @alloc
func.func @alloc() -> !go.ptr<i32> {
    // CHECK: %0 = go.alloc<i32> : !go.ptr<i32>
    %0 = go.alloc<i32> : !go.ptr<i32>
    return %0 : !go.ptr<i32>
}

// CHECK-LABEL: @alloca
func.func @alloca() -> !go.ptr<i32> {
    // CHECK: %0 = go.alloca<i32> : !go.ptr<i32>
    %0 = go.alloca<i32> : !go.ptr<i32>
    return %0 : !go.ptr<i32>
}

// CHECK-LABEL: @load
func.func @load(%arg0 : !go.ptr<i32>) -> i32 {
    // CHECK: %0 = go.load %arg0 : i32
    %0 = go.load %arg0 : i32
    return %0 : i32
}

// CHECK-LABEL: @store
func.func @store(%arg0 : !go.ptr<i32>, %arg1 : i32) {
    // CHECK: go.store %arg1, %arg0 : i32 , !go.ptr<i32>
    go.store %arg1, %arg0 : i32 , !go.ptr<i32>
    return
}

// CHECK-LABEL: @global
func.func @global() -> !go.ptr<i32> {
    // CHECK: %0 = go.global @foo : !go.ptr<i32>
    %0 = "go.global"() <{symbol = "@foo"}> : () -> !go.ptr<i32>
    return %0 : !go.ptr<i32>
}

//===----------------------------------------------------------------------===//
// Struct operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @struct.extract
func.func @struct.extract(%arg0 : !go.struct<{i32, i64}>) -> i64 {
    // CHECK: %0 = go.struct.extract 1, %arg0 : !go.struct<{i32, i64}> -> i64
    %0 = go.struct.extract 1, %arg0 : !go.struct<{i32, i64}> -> i64
    return %0 : i64
}

// CHECK-LABEL: @struct.insert
func.func @struct.insert(%arg0 : !go.struct<{i32, i64}>, %arg1 : i64) -> !go.struct<{i32, i64}> {
    // CHECK: %0 = go.struct.insert %arg1, 1, %arg0 : i64 -> !go.struct<{i32, i64}>
    %0 = go.struct.insert %arg1, 1, %arg0 : i64 -> !go.struct<{i32, i64}>
    return %0 : !go.struct<{i32, i64}>
}

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @constant
func.func @constant() {
    // CHECK: %0 = go.constant(0 : i8) : i8
    %0 = go.constant(0 : i8) : i8

    // CHECK: %1 = go.constant(1 : i16) : i16
    %1 = go.constant(1 : i16) : i16

    // CHECK: %2 = go.constant(2 : i32) : i32
    %2 = go.constant(2 : i32) : i32

    // CHECK: %3 = go.constant(3 : i64) : i64
    %3 = go.constant(3 : i64) : i64

    // CHECK: %4 = go.constant("this is a string") : !go.string
    %4 = go.constant("this is a string") : !go.string

    // CHECK: %5 = go.constant(1.000000e-01 : f32) : f32
    %5 = go.constant(1.000000e-01 : f32) : f32

    // CHECK: %6 = go.constant(3.000000e-01 : f64) : f64
    %6 = go.constant(3.000000e-01 : f64) : f64

    // CHECK: %7 = go.constant(true) : i1
    %7 = go.constant(true) : i1

    // CHECK: %8 = go.constant(false) : i1
    %8 = go.constant (false) : i1

    // CHECK: %9 = go.constant(true) : i1
    %9 = go.constant (1 : i1) : i1

    // CHECK: %10 = go.constant(false) : i1
    %10 = go.constant (0 : i1) : i1

    return
}

// CHECK-LABEL: @constant.struct
func.func @constant.struct(%arg0 : i32, %arg1 : i64) -> !go.struct<{i32, i64}> {
    // CHECK: %0 = go.constant.struct {%arg0, %arg1} : !go.struct<{i32, i64}>
    %0 = go.constant.struct {%arg0, %arg1} : !go.struct<{i32, i64}>
    return %0 : !go.struct<{i32, i64}>
}

// CHECK-LABEL: @constant.zero
func.func @constant.zero() {
    // CHECK: %0 = go.constant.zero : i8
    %0 = go.constant.zero : i8

    // CHECK: %1 = go.constant.zero : i16
    %1 = go.constant.zero : i16

    // CHECK: %2 = go.constant.zero : i32
    %2 = go.constant.zero : i32

    // CHECK: %3 = go.constant.zero : i64
    %3 = go.constant.zero : i64

    // CHECK: %4 = go.constant.zero : f32
    %4 = go.constant.zero : f32

    // CHECK: %5 = go.constant.zero : f64
    %5 = go.constant.zero : f64

    // CHECK: %6 = go.constant.zero : !go.complex64
    %6 = go.constant.zero : !go.complex64

    // CHECK: %7 = go.constant.zero : !go.complex128
    %7 = go.constant.zero : !go.complex128

    // CHECK: %8 = go.constant.zero : !go.struct<{i32, i64}>
    %8 = go.constant.zero : !go.struct<{i32, i64}>

    // CHECK: %9 = go.constant.zero : !go.array<i32[10]>
    %9 = go.constant.zero : !go.array<i32[10]>

    // CHECK: %10 = go.constant.zero : !go.slice<i32>
    %10 = go.constant.zero : !go.slice<i32>

    // CHECK: %11 = go.constant.zero : !go.string
    %11 = go.constant.zero : !go.string

    // CHECK: %12 = go.constant.zero : !go.interface<any>
    %12 = go.constant.zero : !go.interface<any>

    return
}

//===----------------------------------------------------------------------===//
// Casting operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcast
func.func @bitcast(%arg0 : i32, %arg1 : !go.ptr) {
    // CHECK: %0 = go.bitcast %arg0 : i32 -> !go.ptr
    %0 = go.bitcast %arg0 : i32 -> !go.ptr

    // CHECK: %1 = go.bitcast %arg1 : !go.ptr -> !go.ptr<i32>
    %1 = go.bitcast %arg1 : !go.ptr -> !go.ptr<i32>

    return
}

// CHECK-LABEL: @inttoptr
func.func @inttoptr(%arg0 : i8, %arg1 : i16, %arg2 : i32, %arg3 : i64) -> !go.ptr {
    // CHECK: %0 = go.inttoptr %arg0 : i8 -> !go.ptr
    %0 = go.inttoptr %arg0 : i8 -> !go.ptr

    // CHECK: %1 = go.inttoptr %arg1 : i16 -> !go.ptr
    %1 = go.inttoptr %arg1 : i16 -> !go.ptr

    // CHECK: %2 = go.inttoptr %arg2 : i32 -> !go.ptr
    %2 = go.inttoptr %arg2 : i32 -> !go.ptr

    // CHECK: %3 = go.inttoptr %arg3 : i64 -> !go.ptr
    %3 = go.inttoptr %arg3 : i64 -> !go.ptr

    return %0 : !go.ptr
}

// CHECK-LABEL: @ptrtoint
func.func @ptrtoint(%arg0 : !go.ptr, %arg1 : !go.ptr<i32>) -> (i32, i64) {
    // CHECK: %0 = go.ptrtoint %arg0 : !go.ptr -> i32
    %0 = go.ptrtoint %arg0 : !go.ptr -> i32

    // CHECK: %1 = go.ptrtoint %arg1 : !go.ptr<i32> -> i64
    %1 = go.ptrtoint %arg1 : !go.ptr<i32> -> i64

    return %0, %1 : i32, i64
}

// CHECK-LABEL: @ftrunc
func.func @ftrunc(%arg0 : f64) -> f32 {
    // CHECK: %0 = go.ftrunc %arg0 : f64 -> f32
    %0 = go.ftrunc %arg0 : f64 -> f32

    return %0 : f32
}

// CHECK-LABEL: @itrunc
func.func @itrunc(%arg0 : i64) -> i8 {
    // CHECK: %0 = go.itrunc %arg0 : i64 -> i8
    %0 = go.itrunc %arg0 : i64 -> i8

    return %0 : i8
}

// CHECK-LABEL: @fext
func.func @fext(%arg0 : f32) -> f64 {
    // CHECK: %0 = go.fext %arg0 : f32 -> f64
    %0 = go.fext %arg0 : f32 -> f64

    return %0 : f64
}

// CHECK-LABEL: @sext
func.func @sext(%arg0 : i32) -> i64 {
    // CHECK: %0 = go.sext %arg0 : i32 -> i64
    %0 = go.sext %arg0 : i32 -> i64

    return %0 : i64
}

// CHECK-LABEL: @zext
func.func @zext(%arg0 : i32) -> i64 {
    // CHECK: %0 = go.zext %arg0 : i32 -> i64
    %0 = go.zext %arg0 : i32 -> i64

    return %0 : i64
}

// CHECK-LABEL: @ftou
func.func @ftou(%arg0 : f64) -> i64 {
    // CHECK: %0 = go.ftou %arg0 : f64 -> i64
    %0 = go.ftou %arg0 : f64 -> i64

    return %0 : i64
}

// CHECK-LABEL: @ftos
func.func @ftos(%arg0 : f64) -> i64 {
    // CHECK: %0 = go.ftos %arg0 : f64 -> i64
    %0 = go.ftos %arg0 : f64 -> i64

    return %0 : i64
}

// CHECK-LABEL: @utof
func.func @utof(%arg0 : i64) -> f64 {
    // CHECK: %0 = go.utof %arg0 : i64 -> f64
    %0 = go.utof %arg0 : i64 -> f64

    return %0 : f64
}

// CHECK-LABEL: @stof
func.func @stof(%arg0 : i64) -> f64 {
    // CHECK: %0 = go.stof %arg0 : i64 -> f64
    %0 = go.stof %arg0 : i64 -> f64

    return %0 : f64
}

// CHECK-LABEL: @ptrtofunc
func.func @ptrtofunc(%arg0 : !go.ptr) -> (() -> i32) {
    // CHECK: %0 = go.ptrtofunc %arg0 : !go.ptr, () -> i32
    %0 = go.ptrtofunc %arg0 : !go.ptr, () -> i32
    return %0 : () -> i32
}

// CHECK-LABEL: @functoptr
func.func @functoptr(%arg0 : () -> i32) -> !go.ptr {
    // CHECK: %0 = go.functoptr %arg0 : () -> i32, !go.ptr
    %0 = go.functoptr %arg0 : () -> i32, !go.ptr
    return %0 : !go.ptr
}
