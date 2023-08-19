#ifndef GO_OPERATIONS_CONTROL_FLOW_TD
#define GO_OPERATIONS_CONTROL_FLOW_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

def BranchOp : Go_Op<"br", [Terminator, NoMemoryEffect]> {
    let arguments = (ins Variadic<Go_Type>:$destOperands);
    let successors = (successor AnySuccessor:$dest);
    let assemblyFormat = [{
        $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : Go_Op<"cond_br", [Terminator, AttrSizedOperandSegments,
        NoMemoryEffect]> {
     let arguments = (ins I1:$condition,
        Variadic<Go_Type>:$trueDestOperands,
        Variadic<Go_Type>:$falseDestOperands);
     let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
     let assemblyFormat = [{
        $condition `,`
        $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
        $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
        attr-dict
     }];
}

#endif // GO_OPERATIONS_CONTROL_FLOW_TD