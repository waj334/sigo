#ifndef GO_OPERATIONS_STRUCT_TD
#define GO_OPERATIONS_STRUCT_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def ExtractOp : Go_Op<"extract", [Pure, NoMemoryEffect]> {
    let summary = "Aggregate element extraction operator";
    let description = [{
        The `go.extract` operation extracts the value struct element at
        the specific index from the specified aggregate operand and returns the
        value.

        Example:
        ```mlir
        %0 = go.struct.constant {1, 2, 3.0} : !go.struct<{i32, i64, f32}>
        %2 = go.constant (1 : i64) : si32
        %1 = go.extract (%1 : si32) from (%0 : !go.struct<{i32, i64, f32}>) : i64
        ```
    }];
    let arguments = (ins
        I64Attr:$index,
        Go_Aggregate:$aggregate
    );
    let results = (outs Go_Type:$result);
    let assemblyFormat = [{
        $index `from` ` ` `(` $aggregate `:` type($aggregate) `)` attr-dict `:` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

def InsertOp : Go_Op<"insert", [Pure, NoMemoryEffect]> {
    let summary = "Aggregate element insertion operator";
    let description = [{
        The `go.insert` operation inserts the value operand into the
        specified aggregate and returns the resulting struct value.

        Example:
        ```mlir
        %0 = go.constant 3.0 : f32
        %1 = go.struct.constant : !go.struct<{i32, i64, f32}>
        %2 = go.constant (2 : i64) : si32
        %3 = go.insert (%0 : f32) into (%1 : !go.struct<{i32, i64, f32}>) at (%2 : si32) : !go.struct<{i32, i64, f32}>
        ```
    }];
    let arguments = (ins
        Go_Type:$value,
        I64Attr:$index,
        Go_Aggregate:$aggregate
    );
    let results = (outs Go_Aggregate:$result);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` `into` ` ` `(` $aggregate `:` type($aggregate) `)` `at` $index attr-dict `:` type($result)
    }];
    let hasVerifier = 1;
}

#endif // GO_OPERATIONS_STRUCT_TD