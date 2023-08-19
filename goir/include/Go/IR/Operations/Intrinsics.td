#ifndef GO_OPERATIONS_INTRINSICS_TD
#define GO_OPERATIONS_INTRINSICS_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DeclareTypeOp
//===----------------------------------------------------------------------===//

def DeclareTypeOp : Go_Op<"intr.declare_type", [IsolatedFromAbove]> {
    let arguments = (ins
        TypeAttr:$declaredType
    );
    let assemblyFormat = [{
        $declaredType attr-dict-with-keyword
    }];
}

//===----------------------------------------------------------------------===//
// TypeInfoOp
//===----------------------------------------------------------------------===//

def TypeInfoOp : Go_Op<"intr.typeinfo"> {
    let summary = "Type information intrinsic operation";
    let description = [{
        The `go.intr.typeinfo` operation returns the address to reflection
        type information used by the runtime to perform some operations. Type
        information will be generated later and this operation will be replaced
        by a `go.global` operation yielding the address to populated type
        information for the specified type. The result is always an unsafe
        pointer.

        Example:
        ```mlir
        %0 = go.intr.typeinfo !go.slice<i32> : !go.ptr
        ```
    }];
    let arguments = (ins TypeAttr:$T);
    let results = (outs Go_Pointer:$result);
    let assemblyFormat = [{
        `<` $T attr-dict `>` `:` type($result)
    }];
}

#endif // GO_OPERATIONS_INTRINSICS_TD