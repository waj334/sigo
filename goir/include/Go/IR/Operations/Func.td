#ifndef GO_OPERATIONS_FUNC_TD
#define GO_OPERATIONS_FUNC_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Go_Op<"func", [IsolatedFromAbove]> {
    let arguments = (ins
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<SymbolNameAttr>:$sym_name,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);
    let results = (outs Optional<FunctionType>:$result);
    let regions = (region AnyRegion:$body);
    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
    let hasVerifier = 1;
}

#endif // GO_OPERATIONS_FUNC_TD