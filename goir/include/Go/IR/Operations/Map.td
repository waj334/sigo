#ifndef GO_OPERATIONS_MAP_TD
#define GO_OPERATIONS_MAP_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MapUpdateOp
//===----------------------------------------------------------------------===//

def MapUpdateOp : Go_Op<"map.update"> {
    let description = [{
        (%0 : !go.map<!go.i, !go.i>)[0 : !go.i] = (1 : !go.i)
    }];
    let arguments = (ins
        Go_Map:$map,
        Go_Type:$key,
        Go_Type:$value
    );
    let assemblyFormat = [{
        `(` $map `:` type($map) `)` `` `[` $key  `:` type($key) `]` `=` `(` $value `:` type($value) `)` attr-dict-with-keyword
    }];
    let hasCanonicalizeMethod = 1;
}

def MapLookupOp : Go_Op<"map.lookup", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_Map:$map,
        Go_Type:$key
    );
    let results = (outs
        Go_Type:$result,
        Optional<Go_Bool>:$ok
    );
    let assemblyFormat = [{
        $map```[` $key `]` attr-dict-with-keyword `:` functional-type(operands, results)
    }];
    let hasCanonicalizeMethod = 1;
}

#endif // GO_OPERATIONS_MAP_TD