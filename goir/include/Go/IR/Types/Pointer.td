#ifndef GO_TYPES_POINTER_TD
#define GO_TYPES_POINTER_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// Pointer (type constraint only — the C++ type is hand-written)
//===----------------------------------------------------------------------===//

def PointerType : Type<CPred<"::mlir::isa<::mlir::go::PointerType>($_self)">,
                       "Go Pointer Type">;

#endif // GO_TYPES_POINTER_TD
