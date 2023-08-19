#ifndef GO_OP_BASE_TD
#define GO_OP_BASE_TD

include "Go/IR/GoAttrDefs.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Go_Op<string mnemonic, list<Trait> traits = []> :
    Op<Go_Dialect, mnemonic, traits>;

#endif // GO_OP_BASE_TD