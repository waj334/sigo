#ifndef GO_OPERATIONS_CALL_TD
#define GO_OPERATIONS_CALL_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Go_Op<"call", [
            AttrSizedOperandSegments,
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]> {
    let arguments = (ins
        OptionalAttr<FlatSymbolRefAttr>:$callee,
        Optional<Go_Struct>:$callee_value,
        OptionalAttr<TypeAttr>:$signature,
        Variadic<Go_Type>:$call_operands);
    let results = (outs Variadic<Go_Type>);
    let builders = [
        OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::StringRef":$callee, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("callee", SymbolRefAttr::get($_builder.getContext(), callee));
            $_state.addTypes(results);
            llvm::copy(mlir::ArrayRef<int32_t>({0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::StringAttr":$callee, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(results);
            llvm::copy(mlir::ArrayRef<int32_t>({0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::SymbolRefAttr":$callee, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
            llvm::copy(mlir::ArrayRef<int32_t>({0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::TypeAttr":$signature, "::mlir::TypeRange":$results, "::mlir::Value":$callee, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee);
            $_state.addAttribute("signature", signature);
            $_state.addOperands(callee_operands);
            $_state.addTypes(results);
            llvm::copy(mlir::ArrayRef<int32_t>({1, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>
    ];

    let assemblyFormat = [{
        ( `(` $callee_value^ `:` type($callee_value) `)` ) : ( $callee )? `(` $call_operands `)` attr-dict `:` functional-type($call_operands, results)
    }];

    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return callOpResolveOperandTypes(this);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return callOpResolveResultTypes(this);
        }
    }];

    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def CallIndirectOp : Go_Op<"call_indirect"> {
    let arguments = (ins Go_Pointer:$callee, Variadic<Go_Type>:$callee_operands);
    let results = (outs Variadic<Go_Type>:$results);
    let assemblyFormat = [{
        $callee `<` type($callee) `>` `(` $callee_operands `)` attr-dict `:` functional-type($callee_operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// DeferOp
//===----------------------------------------------------------------------===//

def DeferOp : Go_Op<"call_defer", [
            AttrSizedOperandSegments,
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]> {
    let summary = "The defer operation";
    let description = [{
        The `go.defer` operation pushes the function operand to the current
        goroutine's defer stack to be executed before the current function
        returns.

        ```mlir
            %0 = "func.constant"() <{value = @"myfunc"}> : () -> (() -> ())
            "go.defer"(%0) : (() -> ()) -> ()
        ``
    }];
    let arguments = (ins
            Optional<Go_Struct>:$callee_value,
            OptionalAttr<TypeAttr>:$signature,
            OptionalAttr<FlatSymbolRefAttr>:$sym_name,
            Optional<Go_Interface>:$iface_value,
            OptionalAttr<StrAttr>:$method_name,
            Variadic<Go_Type>:$callee_operands);
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "::mlir::StringRef":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", SymbolRefAttr::get($_builder.getContext(), sym_name));
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::StringAttr":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", SymbolRefAttr::get(sym_name));
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::SymbolRefAttr":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", sym_name);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::TypeAttr":$signature, "::mlir::Value":$callee_value, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addAttribute("signature", signature);
            $_state.addOperands(callee_value);
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({1, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::Value":$iface_value, "::mlir::StringAttr":$method_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(iface_value);
            $_state.addAttribute("method_name", method_name);
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 1, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::Value":$iface_value, "::mlir::StringRef":$method_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(iface_value);
            $_state.addAttribute("method_name", SymbolRefAttr::get($_builder.getContext(), method_name));
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 1, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
    ];

    let assemblyFormat = [{
        ( `<` $callee_value^ `:` type($callee_value) `>` )? ( $sym_name^ )? ( `<` $iface_value^ `[` $method_name `]` `:` type($iface_value) `>` )? `(` $callee_operands `)` attr-dict `:` `(` type($callee_operands) `)`
    }];

    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return specialCallOpResolveOperandTypes(this);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return specialCallOpResolveResultTypes(this);
        }
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GoOp
//===----------------------------------------------------------------------===//

def GoOp : Go_Op<"call_go", [
            AttrSizedOperandSegments,
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]> {
    let summary = "Goroutine operation";
    let description = [{
        The `go.go` operation will start a new goroutine that executes the
        specified function with the specified parameters.

        ```mlir
            go.go %0(%1) : (i32)
        ```
    }];
    let arguments = (ins
            Optional<Go_Struct>:$callee_value,
            OptionalAttr<TypeAttr>:$signature,
            OptionalAttr<FlatSymbolRefAttr>:$sym_name,
            Optional<Go_Interface>:$iface_value,
            OptionalAttr<StrAttr>:$method_name,
            Variadic<Go_Type>:$callee_operands);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "::mlir::StringRef":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", SymbolRefAttr::get($_builder.getContext(), sym_name));
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::StringAttr":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", SymbolRefAttr::get(sym_name));
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::SymbolRefAttr":$sym_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(callee_operands);
            $_state.addAttribute("sym_name", sym_name);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::TypeAttr":$signature, "::mlir::Value":$callee_value, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addAttribute("signature", signature);
            $_state.addOperands(callee_value);
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({1, 0, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::Value":$iface_value, "::mlir::StringAttr":$method_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(iface_value);
            $_state.addAttribute("method_name", method_name);
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 1, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
        OpBuilder<(ins "::mlir::Value":$iface_value, "::mlir::StringRef":$method_name, CArg<"::mlir::ValueRange", "{}">:$callee_operands), [{
            $_state.addOperands(iface_value);
            $_state.addAttribute("method_name", SymbolRefAttr::get($_builder.getContext(), method_name));
            $_state.addOperands(callee_operands);
            llvm::copy(mlir::ArrayRef<int32_t>({0, 1, static_cast<int32_t>(callee_operands.size())}), $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
        }]>,
    ];

    let assemblyFormat = [{
        ( `<` $callee_value^ `:` type($callee_value) `>` )? ( $sym_name^ )? ( `<` $iface_value^ `[` $method_name `]` `:` type($iface_value) `>` )? `(` $callee_operands `)` attr-dict `:` `(` type($callee_operands) `)`
    }];

    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return specialCallOpResolveOperandTypes(this);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return specialCallOpResolveResultTypes(this);
        }
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InterfaceCallOp
//===----------------------------------------------------------------------===//

def InterfaceCallOp : Go_Op<"call_interface", [
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]> {
    let summary = "Interface call operation";
    let description = [{
        The `go.icall` operation performs a dynamically dispatched interface
        method call.

        ```mlir
        %0 = go.icall @foo(%interface, %1, %2) : (i32, i32) -> (i8)
        ```
    }];
    let arguments = (ins StrAttr:$callee, Go_Interface:$iface, Variadic<Go_Type>:$callee_operands);
    let results = (outs Variadic<Go_Type>);
    let assemblyFormat = [{
         $callee `<`$iface `:` type($iface) `>` `(` $callee_operands `)` attr-dict `:` functional-type($callee_operands, results)
    }];
    let hasVerifier = 1;

    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return interfaceCallOpResolveOperandTypes(this);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return interfaceCallOpResolveResultTypes(this);
        }
    }];
}

//===----------------------------------------------------------------------===//
// BuiltInCallOp
//===----------------------------------------------------------------------===//

def BuiltInCallOp : Go_Op<"call_builtin"> {
    let arguments = (ins StrAttr:$callee, Variadic<Go_Type>:$callee_operands);
    let results = (outs Variadic<Go_Type>:$callee_results);
    let assemblyFormat = [{
        $callee `(` $callee_operands `)` attr-dict `:` functional-type($callee_operands, $callee_results)
    }];
    let hasVerifier = 1;
    let hasFolder = 1;
}

#endif // GO_OPERATIONS_CALL_TD