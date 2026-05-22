#ifndef GO_OPERATIONS_VALUE_TD
#define GO_OPERATIONS_VALUE_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ZeroOp
//===----------------------------------------------------------------------===//

def ZeroOp : Go_Op<"zero", [Pure, NoMemoryEffect]> {
    let summary = "Zero value of type operation";
    let description = [{
        The `go.zero` operation returns the zero value of the
        specified type.

        ```mlir
        %0 = go.zero : i64
        %1 = go.zero : !go.struct<{i32, i64}>
        ```
    }];
    let hasFolder = 1;
    let results = (outs Go_Type:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// NipOp
//===----------------------------------------------------------------------===//

def NilOp : Go_Op<"nil", [Pure]> {
    let summary = "typed nil value";

    let description = [{
        The `go.nil` operation returns the nil value of the specified nil-capable
        Go type.

        ```mlir
        %0 = go.nil : !go.chan<!go.struct<{}>, RecvOnly>
        %1 = go.nil : !go.ptr<!go.ui32>
        %2 = go.nil : !go.map<!go.string, !go.int>
        %3 = go.nil : !go.slice<!go.i32>
        %4 = go.nil : !go.interface<{}>

        // Named nil-capable types are also valid:
        %5 = go.nil : !go.named<"MyChan", !go.chan<!go.i32>>
        ```
    }];

    let results = (outs Go_Nilable:$result);

    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// ComplexOp
//===----------------------------------------------------------------------===//

def ComplexOp : Go_Op<"complex", [Pure, NoMemoryEffect]> {
    let arguments = (ins Go_Float:$real, Go_Float:$imag);
    let results = (outs Go_Complex:$result);
    let assemblyFormat = [{
        `(` $real `:` type($real) `,` $imag `:` type($imag) `)` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

def ImagOp : Go_Op<"imag", [Pure, NoMemoryEffect]> {
    let arguments = (ins Go_Complex:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

def RealOp : Go_Op<"real", [Pure, NoMemoryEffect]> {
    let arguments = (ins Go_Complex:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// MakeMapOp
//===----------------------------------------------------------------------===//

def MakeMapOp : Go_Op<"make.map", [AlwaysSpeculatable]> {
    let arguments = (ins
        Optional<Go_Integer>:$capacity
    );
    let results = (outs
        Go_Map:$result
    );
    let assemblyFormat = [{
        ( $capacity^ `:` type($capacity) )? attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// MakeSliceOp
//===----------------------------------------------------------------------===//

def MakeSliceOp : Go_Op<"make.slice", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_Integer:$length,
        Optional<Go_Integer>:$capacity
    );
    let results = (outs
        Go_Slice:$result
    );
    let assemblyFormat = [{
        `(` $length `:` type($length) (`,` $capacity^ `:` type($capacity))? `)` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// MakeInterfaceOp
//===----------------------------------------------------------------------===//

def MakeInterfaceOp : Go_Op<"make.interface", [AlwaysSpeculatable]> {
    let arguments = (ins
        TypeAttr:$dynamic_type,
        Go_Pointer:$value
    );
    let results = (outs
        Go_Interface:$result
    );
    let assemblyFormat = [{
        $value attr-dict `:` functional-type(operands, results)
    }];
}

#endif // GO_OPERATIONS_VALUE_TD