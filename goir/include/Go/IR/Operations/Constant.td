#ifndef GO_OPERATIONS_CONSTANT_TD
#define GO_OPERATIONS_CONSTANT_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

def Go_CanBeConstant : AnyAttrOf<[
    ComplexNumberAttr,
    F32Attr,
    F64Attr,
    I1Attr,
    AnyI8Attr,
    AnyI16Attr,
    AnyI32Attr,
    AnyI64Attr,
    StrAttr
]>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Go_Op<"constant", [ConstantLike, Pure, NoMemoryEffect]> {
    let summary = "Zero value of type operation";
    let description = [{
        The `go.constant` operation returns the zero value of the
        specified type.

        ```mlir
        %0 = go.constant (0) : i64
        %1 = go.constant ("this is a string") : !go.string
        %2 = go.constant (0.1) : f32
        %3 = go.constant (1) : i1
        %4 = go.constant (.25i) : !go.complex
        ```
    }];
    let hasFolder = 1;
    let arguments = (ins Go_CanBeConstant:$value);
    let results = (outs Go_Type:$result);
    let assemblyFormat = [{
        `(` $value `)` attr-dict `:` type($result)
    }];
}

#endif // GO_OPERATIONS_CONSTANT_TD