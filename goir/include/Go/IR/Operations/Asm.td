#ifndef GO_OPERATIONS_ASM_TD
#define GO_OPERATIONS_ASM_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

def InlineAsmOp : Go_Op<"inline_asm", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins
        StrAttr:$asm_string,
        TypedArrayAttrBase<AsmConstraintAttr, "register contraints">:$constraints,
        StrArrayAttr:$registerClobbers,
        Variadic<Go_Type>:$operandValues
    );
    let assemblyFormat = [{
        attr-dict `,` $constraints operands `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
}

#endif // GO_OPERATIONS_ASM_TD