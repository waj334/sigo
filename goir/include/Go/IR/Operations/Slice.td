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

#endif // GO_OPERATIONS_SLICE_TD