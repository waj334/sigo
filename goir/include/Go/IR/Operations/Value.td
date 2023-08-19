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

def ZeroOp : Go_Op<"zero", [NoMemoryEffect]> {
    let summary = "Zero value of type operation";
    let description = [{
        The `go.constant.zero` operation returns the zero value of the
        specified type.

        ```mlir
        %0 = go.zero : i64
        %1 = go.zero : !go.struct<{i32, i64}>
        ```
    }];
    //let hasFolder = 1;
    let results = (outs Go_Type:$result);
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
// MakeOp
//===----------------------------------------------------------------------===//

def MakeOp : Go_Op<"make"> {
    let summary = "Make operation";
    let description = [{
        The `go.make` operation creates a value of !go.chan, !go.interface,
        !go.map or !go.slice using the operands. Interfaces are a special case
        in order to abstract away the underlying runtime struct from the IR.

        Example:
        ```mlir
            %0 = go.make : !go.channel<i32>         ; Unbuffered
            %0 = go.make %1 : !go.channel<i32>      ; Buffered

            %0 = go.make %1 : !go.slice<i32>        ; With length
            %0 = go.make %1, %2 : !go.slice<i32>    ; With length and capacity

            %0 = go.make : !go.map<i32,i32>
            %0 = go.make %1 : !go.map<i32,i32>      ; With initial capacity

            %0 = go.make %1 : !go.interface<any>    ; Special case for interfaces
    }];
    let arguments = (ins Variadic<Go_Type>:$args);
    let results = (outs AnyTypeOrNamedOf<[ChanType,
                            SliceType,
                            MapType,
                            InterfaceType]>:$result);
    let assemblyFormat = [{
        $args attr-dict `:` functional-type($args, $result)
    }];
    //let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MakeChanOp
//===----------------------------------------------------------------------===//

def MakeChanOp : Go_Op<"make.chan", [AlwaysSpeculatable]> {
    let arguments = (ins
        Optional<Go_Integer>:$capacity
    );
    let results = (outs
        Go_Chan:$result
    );
    let assemblyFormat = [{
       ( $capacity^ `:` type($capacity) )? attr-dict `:` type($result)
    }];
    let hasCanonicalizeMethod = 1;
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
    let hasCanonicalizeMethod = 1;
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
    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// MakeInterfaceOp
//===----------------------------------------------------------------------===//

def MakeInterfaceOp : Go_Op<"make.interface", [AlwaysSpeculatable]> {
    let arguments = (ins
        TypeAttr:$dynamic_type,
        Go_Type:$value
    );
    let results = (outs
        Go_Interface:$result
    );
    let assemblyFormat = [{
        $value attr-dict `:` functional-type(operands, results)
    }];
    let hasCanonicalizeMethod = 1;
}

#endif // GO_OPERATIONS_VALUE_TD