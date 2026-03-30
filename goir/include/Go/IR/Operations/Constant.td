#ifndef GO_OPERATIONS_CONSTANT_TD
#define GO_OPERATIONS_CONSTANT_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

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

def ConstantOp : Go_Op<"constant", [Pure, NoMemoryEffect]> {
    let summary = "Constant value operation";
    let description = [{
        The `go.constant` operation returns a constant value of the
        specified type.

        ```mlir
        %0 = go.constant (0) : i64
        %1 = go.constant ("this is a string") : !go.string
        %2 = go.constant (0.1) : f32
        %3 = go.constant (1) : i1
        %4 = go.constant (.25i) : !go.complex
        ```
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
    let arguments = (ins
        OptionalAttr<Go_CanBeConstant>:$value,
        OptionalAttr<SymbolNameAttr>:$sym_ref
    );
    let results = (outs Go_Type:$result);
    let regions = (region MaxSizedRegion<1>:$body);
    let assemblyFormat = [{
        ( ` ` `(` $value^ `)` )? ( $sym_ref^ )? attr-dict `:` type($result) ( $body^ )?
    }];
}

def GlobalConstantOp : Go_Op<"constant_global", [Symbol]> {
    let hasVerifier = 1;
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<Go_CanBeConstant>:$value);
    let regions = (region MaxSizedRegion<1>:$body);
    let assemblyFormat = [{
        $sym_name ( `(` $value^ `)` )? ( $body^ )? attr-dict
    }];
}

#endif // GO_OPERATIONS_CONSTANT_TD