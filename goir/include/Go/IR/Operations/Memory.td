#ifndef GO_OPERATIONS_MEMORY_TD
#define GO_OPERATIONS_MEMORY_TD

include "Go/IR/GoOpBase.td"
include "Go/IR/GoEnums.td"
include "Go/IR/GoTypes.td"

include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// alloca
//===----------------------------------------------------------------------===//

def AllocaOp : Go_Op<"alloca", [AlwaysSpeculatable]> {
    let summary = "Stack allocation operation";
    let description = [{
        The `go.alloc` operation allocates memory on the stack or in the heap
        for a value of type `$element`.

        Example:
        ```mlir
        %0 = go.alloca<i32> : !go.ptr<32>
        %1 = go.alloca<i32> heap : !go.ptr<32>
        ```
    }];

    let arguments = (ins
        TypeAttr:$element,
        Optional<Go_Integer>:$numElements,
        OptionalAttr<UnitAttr>:$heap,
        OptionalAttr<StrAttr>:$varName);
    let results = (outs Go_Pointer:$result);
    let assemblyFormat = [{
        (`heap` $heap^)? $element (`[` $numElements^ `:` type($numElements) `]`)? attr-dict-with-keyword `:` type($result)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// load
//===----------------------------------------------------------------------===//

def LoadOp : Go_Op<"load", [AlwaysSpeculatable]> {
    let summary = "Pointer dereference operator";
    let description = [{
        The `go.load` operation performs a deference of the pointer pointer
        operand and returns the result as the pointer's element type.

        Example:
        ```mlir
        %a = go.alloc<i32>
        %0 = go.load %a : i32
    ```
    }];
    let arguments = (ins Go_Pointer:$operand, OptionalAttr<UnitAttr>:$isVolatile, OptionalAttr<UnitAttr>:$isAtomic);
    let results = (outs Go_Type:$result);
    let assemblyFormat = [{
        $operand (`volatile` $isVolatile^)? (`atomic` $isAtomic^)? attr-dict-with-keyword `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// Store
//===----------------------------------------------------------------------===//

def StoreOp : Go_Op<"store"> {
    let summary = "Store operation";
    let description = [{
        The `go.store` operation stores a value at the specified memory
        location.

        Example:
        ```mlir
        go.store %value, %addr : (i32, !go.ptr<i32>)
        ```
    }];

    let arguments = (ins Go_Type:$value, Go_Pointer:$addr, OptionalAttr<UnitAttr>:$isVolatile, OptionalAttr<UnitAttr>:$isAtomic);
    let assemblyFormat = [{
        $value `,` $addr (`volatile` $isVolatile^)? (`atomic` $isAtomic^)? attr-dict-with-keyword `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetElementPointerOp
//===----------------------------------------------------------------------===//

def GetElementPointerOp : Go_Op<"gep", [NoMemoryEffect]> {
    let summary = "Get element pointer operation";
    let description = [{
        The `go.gep` operation calculates the offset within a data structure
        given the element type and returns a pointer to the memory location
        as the result.
    }];
    let arguments = (ins
            Go_Pointer:$value,
            TypeAttr:$baseType,
            Variadic<Go_Integer>:$dynamicIndices,
            DenseI32ArrayAttr:$constIndices);
    let results = (outs Go_Pointer:$result);
    let assemblyFormat = [{
        $value `[` custom<GEPIndices>($dynamicIndices, $constIndices) `]` attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        constexpr static int32_t kValueFlag = 0x80000000;
        constexpr static int32_t kValueIndexMask = 0x7FFFFFFF;
    }];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Go_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["GlobalOp"]>]> {
    let arguments = (ins Go_Type:$initializerValue);
    let assemblyFormat = [{
        $initializerValue attr-dict `:` type($initializerValue)
    }];
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def GlobalOp : Go_Op<"global", [IsolatedFromAbove,
        SingleBlockImplicitTerminator<"YieldOp">, Symbol]> {
    let summary = "Global variable operation";
    let description = [{
        The `go.global` operation creates a named global variable.

        Example:
        ```mlir
        %0 = go.global @foo : !go.ptr<i32>
        ```
    }];
    let arguments = (ins
        TypeAttr:$global_type,
        StrAttr:$sym_name
    );
    let assemblyFormat = [{
        $sym_name attr-dict `:` $global_type ( `init` $initializer^ )?
    }];
    let regions = (region AnyRegion:$initializer);
    let extraClassDeclaration = [{
        /// Return the LLVM type of the global.
        Type getType() {
          return getGlobalType();
        }
        /// Return the initializer region. This may be empty, but if it is not it
        /// terminates in an `llvm.return` op with the initializer value.
        Region &getInitializerRegion() {
          return getOperation()->getRegion(0);
        }
        /// Return the initializer block. If the initializer region is empty this
        /// is nullptr. If it is not nullptr, it terminates with an `llvm.return`
        /// op with the initializer value.
        Block *getInitializerBlock() {
          return getInitializerRegion().empty() ?
            nullptr : &getInitializerRegion().front();
        }

        LogicalResult hasValidInitializer() {
            if (!getInitializerRegion().empty())
            {
                auto& bodyBlock = getInitializerRegion().front();
                for (Operation& op : bodyBlock) {
                    if (failed(operationIsAcceptable(&op))) {
                        return failure();
                    }
                }
                return success();
            }
            return failure();
        }

        static LogicalResult operationIsAcceptable(Operation *op) {
            using namespace LLVM;
            auto iface = mlir::dyn_cast<MemoryEffectOpInterface>(op);
            if (!iface || !iface.hasNoEffect())
            return failure();

            // Only values originating from operations with no side effects are acceptable.
            return success();
        }
      }];
      let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalCtorsOp
//===----------------------------------------------------------------------===//

def GlobalCtorsOp : Go_Op<"global_ctors"> {
    let arguments = (ins
        FlatSymbolRefArrayAttr:$ctors,
        I32ArrayAttr:$priorities
    );
    let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

def AddressOfOp : Go_Op<"addressOf", [Pure, NoMemoryEffect]> {
    let arguments = (ins FlatSymbolRefAttr:$symbol);
    let results = (outs Go_Addressable:$result);
    let assemblyFormat = "$symbol attr-dict `:` type($result)";
}

#endif // GO_OPERATIONS_MEMORY_TD