// RUN: go-opt -allow-unregistered-dialect %s -split-input-file | go-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @array
func.func @array() {
    // CHECK: !go.array<i32[10]>
    "some.op"() : () -> !go.array<i32[10]>

    // CHECK: !go.array<!go.chan<i32>[10]>
    "some.op"() : () -> !go.array<!go.chan<i32>[10]>

    // CHECK: !go.array<!go.complex64[10]>
    "some.op"() : () -> !go.array<!go.complex64[10]>

    // CHECK: !go.array<!go.complex128[10]>
    "some.op"() : () -> !go.array<!go.complex128[10]>

    // CHECK: !go.array<!go.map<i32, i32>[10]>
    "some.op"() : () -> !go.array<!go.map<i32,i32>[10]>

    // CHECK: !go.array<!go.ptr[10]>
    "some.op"() : () -> !go.array<!go.ptr[10]>

    // CHECK: !go.array<!go.ptr<i32>[10]>
    "some.op"() : () -> !go.array<!go.ptr<i32>[10]>

    // CHECK: !go.array<!go.slice<i32>[10]>
    "some.op"() : () -> !go.array<!go.slice<i32>[10]>

    // CHECK: !go.array<!go.string[10]>
    "some.op"() : () -> !go.array<!go.string[10]>

    return
}

// CHECK-LABEL: @chan
func.func @chan() {
    // CHECK: !go.chan<i32>
    "some.op"() : () -> !go.chan<i32>

    // CHECK: !go.chan<i32>
    "some.op"() : () -> !go.chan<i32, SendRecv>

    // CHECK: !go.chan<i32, SendOnly>
    "some.op"() : () -> !go.chan<i32, SendOnly>

    // CHECK: !go.chan<i32, RecvOnly>
    "some.op"() : () -> !go.chan<i32, RecvOnly>

    return
}

// CHECK-LABEL: @complex
func.func @complex() {
    // CHECK: !go.complex64
    "some.op"() : () -> !go.complex64

    //CHECK: !go.complex128
    "some.op"() : () -> !go.complex128

    return
}

// CHECK-LABEL: @interface
func.func @interface() {
    // CHECK: !go.interface<any>
    "some.op"() : () -> !go.interface<any>

    // CHECK: !go.interface<{"f0" = (i32, i32) -> (i32, i16)}>
    "some.op"() : () -> !go.interface<{"f0" = (i32, i32) -> (i32, i16)}>

    // CHECK: !go.interface<{"f1" = (i32, i32) -> (i32, i16), "f2" = () -> ()}>
    "some.op"() : () -> !go.interface<{"f1" = (i32, i32) -> (i32, i16), "f2" = () -> ()}>

    // CHECK: !go.interface<"floop", any>
    "some.op"() : () -> !go.interface<"floop", any>

    // CHECK: !go.interface<"foobar", {"foo" = (!go.interface<"foobar">) -> i32}>
    "some.op"() : () -> !go.interface<"foobar", {"foo" = (!go.interface<"foobar">) -> i32}>

    return
}

// CHECK-LABEL: @map
func.func @map() {
    // CHECK: !go.map<!go.string, i32>
    "some.op"() : () -> !go.map<!go.string, i32>

    return
}

// CHECK-LABEL: @pointer
func.func @pointer() {
    // CHECK: !go.ptr
    "some.op"() : () -> (!go.ptr)

    // CHECK: !go.ptr<i32>
    "some.op"() : () -> (!go.ptr<i32>)

    return
}

// CHECK-LABEL: @slice
func.func @slice() {
    // CHECK: !go.slice<i32>
    "some.op"() : () -> (!go.slice<i32>)

    // CHECK: !go.slice<!go.array<i32[10]>>
    "some.op"() : () -> (!go.slice<!go.array<i32[10]>>)

    return
}

// CHECK-LABEL: @string
func.func @string() {
    // CHECK: !go.string
    "some.op"() : () -> (!go.string)
}

// CHECK-LABEL: @struct
func.func @struct() {
    // CHECK: !go.struct<{i8, i16, i32}>
    "some.op"() : () -> !go.struct<{i8, i16, i32}>

    // CHECK: !go.struct<{i8, i16, i32, !go.chan<i32>}>
    "some.op"() : () -> !go.struct<{i8, i16, i32, !go.chan<i32>}>

    // CHECK: !go.struct<"test-named", {i8, i16, i32, !go.struct<"test-named">}>
    "some.op"() : () -> !go.struct<"test-named", {i8, i16, i32, !go.struct<"test-named">}>

    return
}
