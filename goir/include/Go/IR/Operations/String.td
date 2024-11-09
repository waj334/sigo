#ifndef GO_OPERATIONS_STRING_TD
#define GO_OPERATIONS_STRING_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// StringAddrOp
//===----------------------------------------------------------------------===//

def StringAddrOp : Go_Op<"string.addr", [AlwaysSpeculatable]> {
    let arguments = (ins
        Go_String:$value,
        Go_Integer:$index
    );
    let results = (outs
        Go_Pointer:$result
    );
    let assemblyFormat = [{
        `(` $value `:` type($value) `)` `` `[` $index  `:` type($index) `]` attr-dict-with-keyword `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// StringRangeOp
//===----------------------------------------------------------------------===//

def StringRangeOp : Go_Op<"string.range", [
        DeclareOpInterfaceMethods<RangeOpInterface, [
            "getBodyBlock", "getExitBlock", "getValue", "verifyBodyBlockArgs"]>,
        Pure, Terminator]> {
    let arguments = (ins
        Go_String:$value
    );
    let successors = (successor AnySuccessor:$bodyDest, AnySuccessor:$exitDest);
    let assemblyFormat = [{
        $value `,` $bodyDest `,` $exitDest attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDefinition = [{
        ::mlir::Block* StringRangeOp::getBodyBlock() { return this->getBodyDest(); }
        ::mlir::Block* StringRangeOp::getExitBlock() { return this->getExitDest(); }
    }];
}

#endif // GO_OPERATIONS_STRING_TD