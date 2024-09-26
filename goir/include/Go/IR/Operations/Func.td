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

def FuncOp : Go_Op<"func", [
        AffineScope, AutomaticAllocationScope,
        DeclareOpInterfaceMethods<FunctionOpInterface, [ "verifyBody" ]>, IsolatedFromAbove, OpAsmOpInterface]> {
    let arguments = (ins
        TypeAttrOf<GoFunctionType>:$function_type,
        SymbolNameAttr:$sym_name,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region AnyRegion:$body);
    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "go"; }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Go_Op<"return", [Pure, HasParent<"FuncOp">, ReturnLike, Terminator]> {
    let arguments = (ins
        Variadic<Go_Type>:$operands);
    let builders = [
        OpBuilder<(ins), [{build($_builder, $_state, std::nullopt);}]>
    ];
    let assemblyFormat = "attr-dict ( $operands^ `:` type($operands) )?";
}

#endif // GO_OPERATIONS_FUNC_TD