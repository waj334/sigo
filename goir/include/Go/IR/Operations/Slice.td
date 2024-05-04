#ifndef GO_OPERATIONS_SLICE_TD
#define GO_OPERATIONS_SLICE_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SliceAddrOp
//===----------------------------------------------------------------------===//

def SliceAddrOp : Go_Op<"slice.addr", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_Slice:$slice,
        Go_Integer:$index
    );
    let results = (outs
        Go_Pointer:$result
    );
    let assemblyFormat = [{
        `(` $slice `:` type($slice) `)` `` `[` $index  `:` type($index) `]` attr-dict-with-keyword `:` type($result)
    }];
    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

def SliceOp : Go_Op<"slice", [DeclareOpInterfaceMethods<InferIntRangeInterface>, AttrSizedOperandSegments]> {
    let summary = "Slice operation";
    let description = [{
        The `go.slice` operation yields a slice of an existing string, slice
        or *array X between optional integer bounds Low and High.

        Example:
        ```milr
            %0 = go.slice %1[%2:] : (!go.slice, i32) -> (!go.slice<i32>)
            %0 = go.slice %1[%2:%3] : (!go.slice, i32, i32) -> (!go.slice<i32>)
            %0 = go.slice %1[:%2] : (!go.slice, i32) -> (!go.slice<i32>)
            %0 = go.slice %1[%2:%3:%4] : (!go.slice, i32, i32, i32) -> (!go.slice<i32>)
            %0 = go.slice %1[%2:%3] : (!go.string, i32, i32) -> (!go.slice<i32>)
            %0 = go.slice %1[%2:%3] : (!go.ptr<!go.array<i32[8]>>, i32, i32) -> (!go.slice<i32>)
        ```
    }];
    let arguments = (ins
        AnyTypeOrNamedOf<[Go_Array,Go_Slice,Go_String,Go_Pointer]>:$input,
        Optional<Go_Integer>:$low,
        Optional<Go_Integer>:$high,
        Optional<Go_Integer>:$max);
    let results = (outs AnyTypeOrNamedOf<[Go_Slice, Go_String]>:$result);
    let assemblyFormat = [{
        `(` $input `:` type($input) `)````[```( `(` $low^`:` type($low) `)` )?```:```( `(` $high^ `:` type($high) `)` )? (```:````(` $max^ `:` type($max) `)` )?```]` attr-dict-with-keyword `:` type($result)
    }];
    //let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
}

#endif // GO_OPERATIONS_SLICE_TD