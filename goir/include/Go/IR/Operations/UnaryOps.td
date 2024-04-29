#ifndef GO_OPERATIONS_UNARYOPS_TD
#define GO_OPERATIONS_UNARYOPS_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

class Go_UnaryOp<string mnemonic, list<Trait> traits = []> :
        Go_Op<mnemonic, traits # []> {
    let assemblyFormat = "$operand attr-dict `:` `(` type($operand) `,` type($result) `)`";
}

class Go_BoolUnaryOp<string mnemonic, list<Trait> traits = []> :
    Go_UnaryOp<mnemonic, traits #
        [DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins Go_Bool:$operand)>,
    Results<(outs Go_Bool:$result)>;

class Go_IntUnaryOp<string mnemonic, list<Trait> traits = []> :
    Go_UnaryOp<mnemonic, traits #
        [DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins Go_Integer:$operand)>,
    Results<(outs Go_Integer:$result)>;

class Go_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    Go_UnaryOp<mnemonic, traits # []>,
    Arguments<(ins Go_Float:$operand)>,
    Results<(outs Go_Float:$result)>;

class Go_ComplexUnaryOp<string mnemonic, list<Trait> traits = []> :
    Go_UnaryOp<mnemonic, traits # []>,
    Arguments<(ins Go_Complex:$operand)>,
    Results<(outs Go_Complex:$result)>;

//===----------------------------------------------------------------------===//
// comp (XOR)
//===----------------------------------------------------------------------===//

def ComplementOp : Go_IntUnaryOp<"comp", [NoMemoryEffect]> {
    let summary = "Bitwise complement operator";
    let description = [{
        The `go.comp` operation returns the bitwise complement of the integer
        operand and returns the result.

        Example:
        ```mlir
        %a = go.constant 1 : i32
        %0 = go.comp %a : i32
    ```
    }];
}

//===----------------------------------------------------------------------===//
// negc
//===----------------------------------------------------------------------===//

def NegCOp : Go_ComplexUnaryOp<"negc", [NoMemoryEffect]> {
    let summary = "Complex number negation operation";
    let description = [{
        The `go.negc` negates the complex operand and returns the
        result.

        Example:
        ```mlir
        %a = go.constant 1.0 : !go.complex64
        %0 = go.negc %a : !go.complex64
        ```
    }];
}

//===----------------------------------------------------------------------===//
// negf
//===----------------------------------------------------------------------===//

def NegFOp : Go_FloatUnaryOp<"negf", [NoMemoryEffect]> {
    let summary = "Floating-point negation operation";
    let description = [{
        The `go.negf` negates the floating-point operand and returns the
        result.

        Example:
        ```mlir
        %a = go.constant 1.0 : f32
        %0 = go.negf %a : f32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// negi
//===----------------------------------------------------------------------===//

def NegIOp : Go_IntUnaryOp<"negi", [NoMemoryEffect]> {
    let summary = "Integer negation operation";
    let description = [{
        The `go.negi` negates the integer operand and returns the result. The
        integer operand is interpreted as a signed integer.

        Example:
        ```mlir
        %a = go.constant 1 : i32
        %0 = go.negi %a : i32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// not
//===----------------------------------------------------------------------===//

def NotOp : Go_BoolUnaryOp<"not", [NoMemoryEffect]> {
    let summary = "Logical negation operator";
    let description = [{
        The `go.not` operation performs a logical negation of the operand and
        returns the result.

        Example:
        ```mlir
        %a = go.constant 1 : i1
        %0 = go.not %a : i1
    ```
    }];
}

//===----------------------------------------------------------------------===//
// recv
//===----------------------------------------------------------------------===//

def RecvOp : Go_Op<"recv", [AlwaysSpeculatable]>,
        Arguments<(ins ChanType:$operand, OptionalAttr<UnitAttr>:$commaOk)>,
        Results<(outs Go_Type:$result)> {
    let summary = "Channel receiver operator";
    let description = [{
        The `go.recv` operation receives a value from the channel operand.

        Example:
        ```mlir
        %0 = go.recv %a : i32
    ```
    }];

    let hasCustomAssemblyFormat = 1;
}

#endif // GO_OPERATIONS_UNARYOPS_TD