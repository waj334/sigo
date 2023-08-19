#ifndef GO_OPERATIONS_CASTING_TD
#define GO_OPERATIONS_CASTING_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

def BitcastOp : Go_Op<"bitcast", [Pure, NoMemoryEffect]> {
    let summary = "Bitcast conversion operation";
    let description = [{
        The `go.bitcast` operation converts the input operand to a value of the
        specified type only if both types have the same size in memory.

        ```mlir
        %0 = go.constant(0x20002040) : i32
        %1 = go.inttoptr %0 : !go.ptr
        %2 = go.bitcast %1 : !go.ptr<i32>
        ```
    }];
    let arguments = (ins Go_Type:$value);
    let results = (outs Go_Type:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//

def IntToPtrOp : Go_Op<"inttoptr", [Pure]> {
    let summary = "Integer to pointer conversion operation";
    let description = [{
        The `go.inttoptr` operation converts the integer operand to an opaque
        pointer and returns the value.

        ```mlir
        %0 = go.constant(0x20002040) : i32
        %1 = go.inttoptr %0 : !go.ptr
        ```
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Pointer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//

def PtrToIntOp : Go_Op<"ptrtoint", [Pure]> {
    let summary = "Pointer to integer conversion operation";
    let description = [{
        The `go.ptrtoint` operation converts the pointer operand to an integer
        and returns the value.

        ```mlir
        %0 = go.constant(0x20002040) : i32
        %1 = go.inttoptr %0 : !go.ptr
        %2 = go.ptrtoint %1 : i32
        ```
    }];
    let arguments = (ins Go_Pointer:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// FloatTruncateOp
//===----------------------------------------------------------------------===//

def FloatTruncateOp : Go_Op<"ftrunc", [Pure]> {
    let summary = "Floating-point truncate operation";
    let description = [{
        The `go.ftrunc` operation converts a higher bitwidth floating-point
        value to a lower bitwidth floating-point value.

        ```mlir
        %0 = go.constant(9.9) : f64
        %1 = go.ftrunc %0 : f64 -> f32
    }];
    let arguments = (ins Go_Float:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// IntTruncateOp
//===----------------------------------------------------------------------===//

def IntTruncateOp : Go_Op<"itrunc", [Pure]> {
    let summary = "Integer truncate operation";
    let description = [{
        The `go.itrunc` operation converts a higher bitwidth integer to a
        lower bitwidth integer.

        ```mlir
        %0 = go.constant(99) : i32
        %1 = go.itrunc %0 : i32 -> i16
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// FloatExtendOp
//===----------------------------------------------------------------------===//

def FloatExtendOp : Go_Op<"fext", [Pure]> {
    let summary = "Floating-point extend operation";
    let description = [{
        The `go.fext` operation extends the bitwidth of the floating-point
        operand.

        ```mlir
        %0 = go.constant(0.2) : f32
        %1 = go.fext %0 : f32 -> f64
    }];
    let arguments = (ins Go_Float:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// SignedExtendOp
//===----------------------------------------------------------------------===//

def SignedExtendOp : Go_Op<"sext", [Pure, NoMemoryEffect]> {
    let summary = "Signed integer extend operation";
    let description = [{
        The `go.sext` operation extends the bitwidth of the operand by copying
        appending the sign bit until the bitwidth matches that of the result
        integer type. The operand will be interpreted as a signed integer.

        ```mlir
        %0 = go.constant(99) : i32
        %1 = go.sext %0 : i32 -> i32
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// ZeroExtendOp
//===----------------------------------------------------------------------===//

def ZeroExtendOp : Go_Op<"zext", [Pure, NoMemoryEffect]> {
    let summary = "Unsigned integer integer extend operation";
    let description = [{
        The `go.zext` operation extends the bitwidth of the value operand by
        appending zeros until the value reaches the bitwidth of the result type.
        The operand will be interpreted as an unsigned integer.

        ```mlir
        %0 = go.constant(99) : i32
        %1 = go.zext %0 : i32 -> i32
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// FloatToUnsignedIntOp
//===----------------------------------------------------------------------===//

def FloatToUnsignedIntOp : Go_Op<"ftoui", [Pure, NoMemoryEffect]> {
    let summary = "Floating-point to unsigned integer conversion operation";
    let description = [{
        The `go.ftou` operation converts the floating-point integer operand to
        an unsigned integer value.

        ```mlir
        %0 = go.constant(0.1) : f32
        %1 = go.ftoui %0 : f32 -> i32
        ```
    }];
    let arguments = (ins Go_Float:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// FloatToSignedIntOp
//===----------------------------------------------------------------------===//

def FloatToSignedIntOp : Go_Op<"ftosi", [Pure, NoMemoryEffect]> {
    let summary = "Floating-point to signed integer conversion operation";
    let description = [{
        The `go.ftos` operation converts the floating-point integer operand to
        a signed integer value.

        ```mlir
        %0 = go.constant(0.1) : f32
        %1 = go.ftosi %0 : f32 -> i32
        ```
    }];
    let arguments = (ins Go_Float:$value);
    let results = (outs Go_Integer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// UnsignedIntToFloatOp
//===----------------------------------------------------------------------===//

def UnsignedIntToFloatOp : Go_Op<"uitof", [Pure, NoMemoryEffect]> {
    let summary = "Unsigned integer to floating-point conversion operation";
    let description = [{
        The `go.utof` operation converts the unsigned integer operand to
        a floating-point value.

        ```mlir
        %0 = go.constant(1) : i32
        %1 = go.uitof %0 : i32 -> f32
        ```
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// SignedIntToFloatOp
//===----------------------------------------------------------------------===//

def SignedIntToFloatOp : Go_Op<"sitof", [Pure, NoMemoryEffect]> {
    let summary = "Signed integer to floating-point conversion operation";
    let description = [{
        The `go.stof` operation converts the signed integer operand to
        a floating-point value.

        ```mlir
        %0 = go.constant(-1) : i32
        %1 = go.sitof %0 : i32 -> f64
        ```
    }];
    let arguments = (ins Go_Integer:$value);
    let results = (outs Go_Float:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// FunctionToPointerOp
//===----------------------------------------------------------------------===//

def FunctionToPointerOp : Go_Op<"functoptr", [Pure, NoMemoryEffect]> {
    let summary = "Function to pointer operation";
    let description = [{
        The `go.functoptr` operation converts the function operand to an opaque
        pointer and then returns the value.

        ```mlir
        %0 = func.constant @foo : () -> i32
        %1 = go.functoptr %1 : () -> i32, go.ptr
        ```
    }];
    let arguments = (ins Go_Function:$value);
    let results = (outs Go_Pointer:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `,` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PointerToFunctionOp
//===----------------------------------------------------------------------===//

def PointerToFunctionOp : Go_Op<"ptrtofunc", [Pure, NoMemoryEffect]> {
    let summary = "Pointer to function conversion operation";
    let description = [{
        The `go.ptrtofunc` operation converts the pointer operand and returns
        a function with the specified signature. The result is effectively a
        function pointer.

        ```mlir
        %0 = func.constant @foo : () -> i32
        %1 = go.functoptr %1 : () -> i32, go.ptr
        %2 = go.ptrtofunc $1 : go.ptr, () -> i32
        ```
    }];
    let arguments = (ins Go_Pointer:$value);
    let results = (outs Go_Function:$result);
    let assemblyFormat  = [{
        $value attr-dict-with-keyword `:` type($value) `,` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ChangeInterfaceOp
//===----------------------------------------------------------------------===//

def ChangeInterfaceOp : Go_Op<"change", [AlwaysSpeculatable]> {
    let summary = "Change interface operation";
    let description = [{
        The `go.change` operation constructs a value of one interface type from
        a value of another interface type known to be assignable to it.

        ```mlir
        %1 = go.change %0 : !go.interface -> !go.interface<any>
        ```
    }];
    let arguments = (ins Go_Interface:$value);
    let results = (outs Go_Interface:$result);
    let assemblyFormat = [{
        $value attr-dict-with-keyword `:` type($value) `to` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TypeAssertOp
//===----------------------------------------------------------------------===//

def TypeAssertOp : Go_Op<"type.assert", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_Interface:$value
    );
    let results = (outs
        Go_Type:$result,
        Optional<Go_Bool>:$ok
    );
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` attr-dict `:` type(results)
    }];
    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ComplexExtendOp
//===----------------------------------------------------------------------===//

def ComplexExtendOp : Go_Op<"cext", [Pure]> {
    let arguments = (ins Go_Complex:$value);
    let results = (outs Go_Complex:$result);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// ComplexTruncateOp
//===----------------------------------------------------------------------===//

def ComplexTruncateOp : Go_Op<"ctrunc", [Pure]> {
    let arguments = (ins Go_Complex:$value);
    let results = (outs Go_Complex:$result);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` attr-dict `:` type($result)
    }];
}

#endif // GO_OPERATIONS_CASTING_TD