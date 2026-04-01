#ifndef GO_OPERATIONS_BINOPS_TD
#define GO_OPERATIONS_BINOPS_TD

include "Go/IR/GoInterfaces.td"
include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

class Go_BinOp<string mnemonic, list<Trait> traits = []> :
        Go_Op<mnemonic, traits # [
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]> {
    let assemblyFormat = "$lhs `,` $rhs attr-dict-with-keyword `:` functional-type(operands, results)";
    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return binOpResolveTypes(this, 2);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return binOpResolveTypes(this, 1);
        }
    }];
}

class Go_IntBinOp<string mnemonic, list<Trait> traits = []> :
        Go_BinOp<mnemonic, traits #
            [
                DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
            ]>,
        Arguments<(ins Go_Integer:$lhs, Go_Integer:$rhs)>,
        Results<(outs Go_Integer:$result)>;

class Go_FloatBinOp<string mnemonic, list<Trait> traits = []> :
        Go_BinOp<mnemonic, traits #
                [Pure]>,
        Arguments<(ins Go_Float:$lhs, Go_Float:$rhs)>,
        Results<(outs Go_Float:$result)>;

class Go_ComplexBinOp<string mnemonic, list<Trait> traits = []> :
        Go_BinOp<mnemonic, traits #
                [Pure]>,
        Arguments<(ins Go_Complex:$lhs, Go_Complex:$rhs)>,
        Results<(outs Go_Complex:$result)>;

class Go_CompareOp<string mnemonic, list<Trait> traits = []> :
        Go_Op<mnemonic, traits # [
            Pure,
            TypesMatchWith<"result type has i1 element type and same shape as operands", "lhs", "result", "::mlir::go::getI1SameShape($_self)">,
            HasCompatibleOperandsTypes,
            DeclareOpInterfaceMethods<UntypedTypeResolverInterface, ["resolveOperandTypes", "resolveResultTypes"]>
        ]>{
    let results = (outs Go_Bool:$result);
    let assemblyFormat = "$lhs ` ` $predicate ` ` $rhs attr-dict-with-keyword `:` functional-type(operands, results)";
    let extraClassDefinition = [{
        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveOperandTypes() {
            return binOpResolveTypes(this, 2);
        }

        ::mlir::SmallVector<::mlir::Type> $cppClass::resolveResultTypes() {
            return binOpResolveTypes(this, 1);
        }
    }];
}

//===----------------------------------------------------------------------===//
// addc
//===----------------------------------------------------------------------===//

def AddCOp : Go_ComplexBinOp<"addc", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two complex inputs and yields their sum.";
    let description = [{
        The `go.addc` operation takes two complex operands and returns
        the sum of the operands as the result.

        Example:
        ```mlir
        %0 = go.addc %a, %b : !go.complex64
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// addf
//===----------------------------------------------------------------------===//

def AddFOp : Go_FloatBinOp<"addf", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two floating-point inputs and yields their sum.";
    let description = [{
        The `go.addf` operation takes two floating-point operands and returns
        the sum of the operands as the result.

        Example:
        ```mlir
        %0 = go.addf %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// addi
//===----------------------------------------------------------------------===//

def AddIOp : Go_IntBinOp<"addi", [Pure, Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two integer inputs and yields their sum.";
    let description = [{
        The `go.addi` operation takes two integer operands and returns the sum
        of the operands as the result.

        Example:
        ```mlir
        %0 = go.addi %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// addstr
//===----------------------------------------------------------------------===//

def AddStrOp : Go_BinOp<"addstr", [Pure, HasCompatibleOperandsAndResultTypes]>,
        Arguments<(ins Go_String:$lhs, Go_String:$rhs)>,
        Results<(outs Go_String:$result)>{
    let summary = "Accepts two string inputs and concatenates them.";
    let description = [{
        The `go.addstr` operation takes two string operands and returns the
        concatenation of the operands as the result.

        Example:
        ```mlir
        %0 = go.addstr %a, %b : !go.string
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// and
//===----------------------------------------------------------------------===//

def AndOp : Go_IntBinOp<"and", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two operands and indicates whether both are true.";
    let description = [{
        The `go.and` operation takes two operands, performs a bitwise and
        operation and returns the result.

        Example:
        ```mlir
        %0 = go.and %a, %b : i32
        ```
    }];

    let arguments = (ins Go_IntegerLike:$lhs, Go_IntegerLike:$rhs);
    let results = (outs Go_IntegerLike:$result);
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// andnot
//===----------------------------------------------------------------------===//

def AndNotOp : Go_IntBinOp<"andnot", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two operands and effectively performs a bit clear";
    let description = [{
        The `go.andnot` operation takes two operands and returns the result of
        a bitwise and of the first operand and the complement of the second
        operand.

        Example:
        ```mlir
        %0 = go.andnot %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// cmpc
//===----------------------------------------------------------------------===//

def CmpCOp : Go_CompareOp<"cmpc", [NoMemoryEffect]> {
    let summary = "Accepts two complex inputs and compares them.";
    let description = [{
        The `go.cmpc` operation takes two complex operands, compares
        them and returns an i1 result.

        Example:
        ```mlir
        %a = go.constant 1 : !go.complex64
        %b = go.constant 2 : !go.complex64
        %0 = go.cmpf ge, %a, %b : i1
        ```
    }];
    let arguments = (ins Go_CmpFPredicate:$predicate,
                           Go_Complex:$lhs,
                           Go_Complex:$rhs);
    let hasVerifier = 1;
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// cmpf
//===----------------------------------------------------------------------===//

def CmpFOp : Go_CompareOp<"cmpf", [NoMemoryEffect]> {
    let summary = "Accepts two floating-point inputs and compares them.";
    let description = [{
        The `go.cmpf` operation takes two floating-point operands, compares
        them and returns an i1 result.

        Example:
        ```mlir
        %a = go.constant 1 : f32
        %b = go.constant 2 : f32
        %0 = go.cmpf ge, %a, %b : i1
        ```
    }];
    let arguments = (ins Go_CmpFPredicate:$predicate,
                           Go_Float:$lhs,
                           Go_Float:$rhs);
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// cmpi
//===----------------------------------------------------------------------===//

def CmpIOp : Go_CompareOp<"cmpi", [NoMemoryEffect]> {
    let summary = "Accepts two integer inputs and compares them.";
    let description = [{
        The `go.cmpi` operation takes two integer operands, compares
        them and returns an i1 result.

        Example:
        ```mlir
        %a = go.constant 1 : i32
        %b = go.constant 2 : i32
        %0 = go.cmpi ge, %a, %b : i1
        ```
    }];
    let arguments = (ins Go_CmpIPredicate:$predicate,
                               Go_IntegerLike:$lhs,
                               Go_IntegerLike:$rhs);
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// cmp.interface
//===----------------------------------------------------------------------===//

def CmpInterfaceOp : Go_BinOp<"cmp.interface", [Commutative]> {
    let arguments = (ins
        Go_Type:$lhs,
        Go_Type:$rhs
    );
    let results = (outs
        Go_Bool:$result
    );
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// cmp.string
//===----------------------------------------------------------------------===//

def CmpStringOp : Go_BinOp<"cmp.string", [Commutative]> {
    let arguments = (ins
        Go_CmpPredicate:$predicate,
        Go_String:$lhs,
        Go_String:$rhs
    );
    let results = (outs
        Go_Bool:$result
    );
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// divc
//===----------------------------------------------------------------------===//

def DivCOp : Go_ComplexBinOp<"divc", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two signed complex inputs and yields their quotient.";
    let description = [{
        The `go.divc` operation takes two complex operands and returns
        the quotient of the operands as the result.

        Example:
        ```mlir
        %0 = go.divc %a, %b : !go.complex64
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// divf
//===----------------------------------------------------------------------===//

def DivFOp : Go_FloatBinOp<"divf", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two signed floating-point inputs and yields their quotient.";
    let description = [{
        The `go.divf` operation takes two floating-point operands and returns
        the quotient of the operands as the result.

        Example:
        ```mlir
        %0 = go.divf %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// divsi
//===----------------------------------------------------------------------===//

def DivSIOp : Go_IntBinOp<"divsi", [Pure, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two signed integer inputs and yields their quotient.";
    let description = [{
        The `go.divsi` operation takes two signed integer operands and returns
        the quotient of the operands as the result.

        Example:
        ```mlir
        %0 = go.divsi %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// divui
//===----------------------------------------------------------------------===//

def DivUIOp : Go_IntBinOp<"divui", [Pure, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two unsigned integer inputs and yields their quotient.";
    let description = [{
        The `go.divui` operation takes two integer operands and returns the
        quotient of the operands as the result.

        Example:
        ```mlir
        %0 = go.divui %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// mulc
//===----------------------------------------------------------------------===//

def MulCOp : Go_ComplexBinOp<"mulc", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two complex inputs and yields their product.";
    let description = [{
        The `go.mulc` operation takes two complex operands and
        returns the product of the operands as the result.

        Example:
        ```mlir
        %0 = go.mulf %a, %b : !go.complex64
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// mulf
//===----------------------------------------------------------------------===//

def MulFOp : Go_FloatBinOp<"mulf", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two floating-point inputs and yields their product.";
    let description = [{
        The `go.mulf` operation takes two floating-point operands and
        returns the product of the operands as the result.

        Example:
        ```mlir
        %0 = go.mulf %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// muli
//===----------------------------------------------------------------------===//

def MulIOp : Go_IntBinOp<"muli", [Pure, Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two integer inputs and yields their product.";
    let description = [{
        The `go.muli` operation takes two unsigned integer operands and
        returns the product of the operands as the result.

        Example:
        ```mlir
        %0 = go.muli %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// or
//===----------------------------------------------------------------------===//

def OrOp : Go_IntBinOp<"or", [Commutative, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two operands and indicates whether either are true.";
    let description = [{
        The `go.or` operation takes two operands, performs a bitwise or and
        returns the result.

        Example:
        ```mlir
        %0 = go.or %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// remf
//===----------------------------------------------------------------------===//

def RemFOp : Go_FloatBinOp<"remf", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Floating-point division remainder operation.";
    let description = [{
        The `go.remf` operation takes two operands and returns the remainder
        of the first operand divided by the second operand.

        Example:
        ```mlir
        %0 = go.remf %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// remsi
//===----------------------------------------------------------------------===//

def RemSIOp : Go_IntBinOp<"remsi", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Signed integer division remainder operation.";
    let description = [{
        The `go.remsi` operation takes two operands and returns the remainder
        of the first operand divided by the second operand. The integer is
        interpreted as a signed integer.

        Example:
        ```mlir
        %0 = go.remsi %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// remui
//===----------------------------------------------------------------------===//

def RemUIOp : Go_IntBinOp<"remui", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Unsigned integer division remainder operation.";
    let description = [{
        The `go.remui` operation takes two operands and returns the remainder
        of the first operand divided by the second operand. The integer is
        interpreted as an unsigned integer.

        Example:
        ```mlir
        %0 = go.remui %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// shl
//===----------------------------------------------------------------------===//

def ShlOp : Go_IntBinOp<"shl", [NoMemoryEffect]> {
    let summary = "Shifts an integer value to the left by a variable amount.";
    let description = [{
        The `go.shl` operation shifts an integer value by a variable amount. The
        low-order bits are filled zeros.

        Example:
        ```mlir
        %0 = go.shl %a, %b : i32
        ```
    }];
    let assemblyFormat = "$lhs `,` $rhs attr-dict-with-keyword `:` functional-type(operands, results)";
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// shru
//===----------------------------------------------------------------------===//

def ShrUIOp : Go_IntBinOp<"shrui", [NoMemoryEffect]> {
    let summary = "Shifts an integer value to the right by a variable amount.";
    let description = [{
        The `go.shl` operation shifts an integer value by a variable amount. The
        low-order bits are filled zeros. The integer value is interpreted as
        unsigned.

        Example:
        ```mlir
        %0 = go.shru %a, %b : i32
        ```
    }];
    let assemblyFormat = "$lhs `,` $rhs attr-dict-with-keyword `:` functional-type(operands, results)";
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// shrsi
//===----------------------------------------------------------------------===//

def ShrSIOp : Go_IntBinOp<"shrsi", [NoMemoryEffect]> {
    let summary = "Shifts an integer value to the right by a variable amount.";
    let description = [{
        The `go.shl` operation shifts an integer value by a variable amount. The
        low-order bits are filled zeros. The integer value is interpreted as
        signed.

        Example:
        ```mlir
        %0 = go.shrsi %a, %b : i32
        ```
    }];
    let assemblyFormat = "$lhs `,` $rhs attr-dict-with-keyword `:` functional-type(operands, results)";
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// subc
//===----------------------------------------------------------------------===//

def SubCOp : Go_ComplexBinOp<"subc", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two complex inputs and yields their difference.";
    let description = [{
        The `go.subc` operation takes two complex operands and returns the
        difference of the operands as the result.

        Example:
        ```mlir
        %0 = go.subc %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// subf
//===----------------------------------------------------------------------===//

def SubFOp : Go_FloatBinOp<"subf", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two floating-point inputs and yields their difference.";
    let description = [{
        The `go.subf` operation takes two floating-point operands and returns the
        difference of the operands as the result.

        Example:
        ```mlir
        %0 = go.subf %a, %b : f32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// subi
//===----------------------------------------------------------------------===//

def SubIOp : Go_IntBinOp<"subi", [Pure, HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two integer inputs and yields their difference.";
    let description = [{
        The `go.subi` operation takes two integer operands and returns the
        difference of the operands as the result.

        Example:
        ```mlir
        %0 = go.subi %a, %b : i32
        ```
    }];
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// xor
//===----------------------------------------------------------------------===//

def XorOp : Go_IntBinOp<"xor", [HasCompatibleOperandsAndResultTypes,
            NoMemoryEffect]> {
    let summary = "Accepts two operands and indicates whether either are true or false if both are true.";
    let description = [{
        The `go.xor` operation takes two operands and returns 1 if one is
        true or 0 if none or both are true.

        Example:
        ```mlir
        %0 = go.xor %a, %b : i1
        ```
    }];

    let arguments = (ins Go_IntegerLike:$lhs, Go_IntegerLike:$rhs);
    let results = (outs Go_IntegerLike:$result);
    let hasFolder = 1;
}

#endif // GO_OPERATIONS_BINOPS_TD