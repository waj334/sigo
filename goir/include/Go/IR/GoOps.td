#ifndef GO_OPS_TD
#define GO_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "Go/IR/GoAttrDefs.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "Go/IR/Operations/Atomic.td"
include "Go/IR/Operations/BinOps.td"
include "Go/IR/Operations/Builtin.td"
include "Go/IR/Operations/Call.td"
include "Go/IR/Operations/Casting.td"
include "Go/IR/Operations/Constant.td"
include "Go/IR/Operations/ControlFlow.td"
include "Go/IR/Operations/Func.td"
include "Go/IR/Operations/Intrinsics.td"
include "Go/IR/Operations/Map.td"
include "Go/IR/Operations/Memory.td"
include "Go/IR/Operations/Slice.td"
include "Go/IR/Operations/Aggregate.td"
include "Go/IR/Operations/UnaryOps.td"
include "Go/IR/Operations/Value.td"

#endif // GO_OPS_TD