#ifndef GO_OPERATIONS_INTRINSICS_TD
#define GO_OPERATIONS_INTRINSICS_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RunDefersOp
//===----------------------------------------------------------------------===//

def RunDefersOp : Go_Op<"intrinsic.run_defers"> {
    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // GO_OPERATIONS_INTRINSICS_TD