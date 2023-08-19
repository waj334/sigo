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
// CallIndirectOp
//===----------------------------------------------------------------------===//

def CallIndirectOp : Go_Op<"call_indirect", [
            TypesMatchWith<"callee input types match argument types",
                         "callee", "callee_operands",
                         "::mlir::go::cast<FunctionType>($_self).getInputs()">,
            TypesMatchWith<"callee result types match result types",
                         "callee", "results",
                         "::mlir::go::cast<FunctionType>($_self).getResults()">
        ]> {
    let arguments = (ins Go_Function:$callee, Variadic<Go_Type>:$callee_operands);
    let results = (outs Variadic<Go_Type>:$results);
    let assemblyFormat = [{
        $callee `(` $callee_operands `)` attr-dict `:` type($callee)
    }];
}

//===----------------------------------------------------------------------===//
// defer
//===----------------------------------------------------------------------===//

def DeferOp : Go_Op<"call_defer"> {
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
                AnyTypeOrNamedOf<[Go_Function, Go_Interface, Go_Struct]>:$callee,
                OptionalAttr<StrAttr>:$method_name,
                Variadic<Go_Type>:$callee_operands);
    let assemblyFormat = [{
        $callee (`<` $method_name^ `>`)? `(` $callee_operands `)` attr-dict `:` type(operands)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GoOp
//===----------------------------------------------------------------------===//

def GoOp : Go_Op<"call_go"> {
    let summary = "Goroutine operation";
    let description = [{
        The `go.go` operation will start a new goroutine that executes the
        specified function with the specified parameters.

        ```mlir
            go.go %0(%1) : (i32)
        ```
    }];
    let arguments = (ins
            AnyTypeOrNamedOf<[Go_Addressable, Go_Interface, Go_Struct]>:$callee,
            OptionalAttr<StrAttr>:$method_name,
            Variadic<Go_Type>:$callee_operands);
    let assemblyFormat = [{
        $callee (`<` $method_name^ `>`)? `(` $callee_operands `)` attr-dict `:` type(operands)
    }];
    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// InterfaceCallOp
//===----------------------------------------------------------------------===//

def InterfaceCallOp : Go_Op<"call_interface"> {
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
}

//===----------------------------------------------------------------------===//
// RuntimeCallOp
//===----------------------------------------------------------------------===//

def RuntimeCallOp : Go_Op<"call_runtime"> {
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Go_Type>:$callee_operands);
    let results = (outs Variadic<Go_Type>:$callee_results);
    let assemblyFormat = [{
        $callee `(` $callee_operands `)` attr-dict `:` functional-type($callee_operands, $callee_results)
    }];
}

#endif // GO_OPERATIONS_CALL_TD