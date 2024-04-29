#ifndef GO_OPERATIONS_ATOMIC_TD
#define GO_OPERATIONS_ATOMIC_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AtomicAddIOp
//===----------------------------------------------------------------------===//

def AtomicAddIOp : Go_Op<"atomic.addi"> {
    let summary = "Accepts two integer inputs and atomically yields their sum.";
    let description = [{
        The `go.atomic.addi` operation takes two integer operands, atomically calculates their sum and returns the
        result.

        Example:
        ```mlir
        %0 = go.atomic.addi %a, %b : i32
        ```
    }];
    let arguments = (ins Go_Pointer:$addr, Go_Integer:$rhs);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = "`(` $addr `:` type($addr) `)` `,` `(` $rhs `:` type($rhs) `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicSwapIOp
//===----------------------------------------------------------------------===//

def AtomicSwapIOp : Go_Op<"atomic.swap"> {
    let summary = "Atomic swap operation";
    let description = [{
        The `go.atomic.addi` atomically stores new into *addr and returns the previous *addr value.

        Example:
        ```mlir
        %0 = go.atomic.swap %a, %b : i32
        ```
    }];
    let arguments = (ins Go_Pointer:$addr, AnyTypeOrNamedOf<[Go_Integral, Go_Pointer]>:$rhs);
    let results = (outs AnyTypeOrNamedOf<[Go_Integral, Go_Pointer]>:$result);
    let assemblyFormat = "`(` $addr `:` type($addr) `)` `,` `(` $rhs `:` type($rhs) `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicCompareAndSwapIOp
//===----------------------------------------------------------------------===//

def AtomicCompareAndSwapIOp : Go_Op<"atomic.cmpswap"> {
    let summary = "Atomic compare and swap operation";
    let description = [{
        The `go.atomic.cmpswap` atomically stores value into *addr and returns the previous *addr if the value matches
        old.

        Example:
        ```mlir
        %0 = go.atomic.cmpswap %a, %b, %c : i32
        ```
    }];
    let arguments = (ins Go_Pointer:$addr, AnyTypeOrNamedOf<[Go_Integral, Go_Pointer]>:$old, AnyTypeOrNamedOf<[Go_Integral, Go_Pointer]>:$value);
    let results = (outs AnyTypeOrNamedOf<[Go_Integral, Go_Pointer]>:$result);
    let assemblyFormat = "`(` $addr `:` type($addr) `)` `,` `(` $old `:` type($old) `)` `,` `(` $value `:` type($value) `)` attr-dict `:` type($result)";
}

#endif // GO_OPERATIONS_ATOMIC_TD