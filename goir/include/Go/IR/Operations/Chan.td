#ifndef GO_OPERATIONS_CHAN_TD
#define GO_OPERATIONS_CHAN_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ChanRecvOp
//===----------------------------------------------------------------------===//

def ChanRecvOp : Go_Op<"chan.recv"> {
    let description = [{
        Channel receive operation.
    }];

    let arguments = (ins
        Go_Chan: $channel
    );

    let results = (outs
        Go_Type: $result,
        Optional<Go_Bool>:$ok
    );

    let assemblyFormat = [{
        $channel attr-dict-with-keyword `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// ChanSendOp
//===----------------------------------------------------------------------===//

def ChanSendOp : Go_Op<"chan.send"> {
    let description = [{
        Channel send operation.
    }];

    let arguments = (ins
        Go_Chan: $channel,
        Go_Type: $value
    );

    let assemblyFormat = [{
        $channel `,` $value attr-dict-with-keyword `:` `(` type($channel) `,` type($value) `)`
    }];
}

//===----------------------------------------------------------------------===//
// ChanRangeOp
//===----------------------------------------------------------------------===//

def ChanRangeOp : Go_Op<"chan.range", [
        DeclareOpInterfaceMethods<RangeOpInterface, [
            "getBodyBlock", "getExitBlock", "getValue", "verifyBodyBlockArgs"]>,
        Pure, Terminator]> {
    let description = [{
        Channel range operation.
    }];
    let arguments = (ins
        Go_Chan:$channel
    );
    let successors = (successor AnySuccessor:$bodyDest, AnySuccessor:$exitDest);
    let assemblyFormat = [{
        $channel `,` $bodyDest `,` $exitDest attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDefinition = [{
        ::mlir::Block* ChanRangeOp::getBodyBlock() { return this->getBodyDest(); }
        ::mlir::Block* ChanRangeOp::getExitBlock() { return this->getExitDest(); }
    }];
}

//===----------------------------------------------------------------------===//
// ChanSelectOp
//===----------------------------------------------------------------------===//

def ChanSelectOp : Go_Op<"chan.select", [Pure, Terminator]> {
    let arguments = (ins
        UnitAttr:$hasDefault,
        DenseBoolArrayAttr:$send,
        Variadic<Go_Chan>:$channel
    );
    let successors = (successor
        AnySuccessor:$defaultDest,
        AnySuccessor:$exitDest,
        VariadicSuccessor<AnySuccessor>:$caseDests
    );
    let assemblyFormat = [{
        $channel `[` $caseDests `]` $defaultDest `,` $exitDest attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
}

#endif // GO_OPERATIONS_CHAN_TD