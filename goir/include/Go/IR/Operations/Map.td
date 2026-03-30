#ifndef GO_OPERATIONS_MAP_TD
#define GO_OPERATIONS_MAP_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MapAddrOp
//===----------------------------------------------------------------------===//

def MapAddrOp : Go_Op<"map.addr", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_Map:$map,
        Go_Pointer:$key
    );
    let results = (outs
        Go_Pointer:$result
    );
    let assemblyFormat = [{
        $map```[` $key `]` attr-dict-with-keyword `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// MapLookupOp
//===----------------------------------------------------------------------===//

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
}

//===----------------------------------------------------------------------===//
// MapRangeOp
//===----------------------------------------------------------------------===//

def MapRangeOp : Go_Op<"map.range", [
        DeclareOpInterfaceMethods<RangeOpInterface, [
            "getBodyBlock", "getExitBlock", "getValue", "verifyBodyBlockArgs"]>,
        Pure, Terminator]> {
    let arguments = (ins
        Go_Map:$map
    );
    let successors = (successor AnySuccessor:$bodyDest, AnySuccessor:$exitDest);
    let assemblyFormat = [{
        $map `,` $bodyDest `,` $exitDest attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDefinition = [{
        ::mlir::Block* MapRangeOp::getBodyBlock() { return this->getBodyDest(); }
        ::mlir::Block* MapRangeOp::getExitBlock() { return this->getExitDest(); }
    }];
}

#endif // GO_OPERATIONS_MAP_TD