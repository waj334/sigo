#ifndef GO_OPERATIONS_BUILTIN_TD
#define GO_OPERATIONS_BUILTIN_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// panic
//===----------------------------------------------------------------------===//

def PanicOp : Go_Op<"panic", [Terminator]> {
    let summary = "The panic operation";
    let description = [{
        The `go.panic` operation terminates execution of the current function
        and then begins calling the defer stack.

        ```mlir
        go.panic (%0 : !go.interface<any>)
        go.panic (%0 : !go.int) -> [bb2]    ; with branch (to recover block)
        ```
    }];
    let arguments = (ins Go_Interface:$value);
    let successors = (successor VariadicSuccessor<AnySuccessor>:$dest);
    let assemblyFormat = [{
        `(`$value `:` type($value)`)` attr-dict `->` $dest
    }];
}

//===----------------------------------------------------------------------===//
// recover
//===----------------------------------------------------------------------===//

def RecoverOp : Go_Op<"recover"> {
    let summary = "Panic recovery operation";
    let description = [{
        The `go.recover` operation resumes execution following a panic
        effectively handling the panic.

        ```mlir
        %0 = go.recover : !go.interface<any>;
        ```
    }];
    let results = (outs Go_Interface:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

#endif // GO_OPERATIONS_BUILTIN_TD