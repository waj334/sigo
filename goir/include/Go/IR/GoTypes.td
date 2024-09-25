#ifndef GO_TYPES
#define GO_TYPES

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "Go/IR/GoDialect.td"

class GoDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Go_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

include "Go/IR/Types/Array.td"
include "Go/IR/Types/Chan.td"
include "Go/IR/Types/Function.td"
include "Go/IR/Types/Map.td"
include "Go/IR/Types/Pointer.td"
include "Go/IR/Types/Primitives.td"
include "Go/IR/Types/Slice.td"
include "Go/IR/Types/String.td"

def NamedType : GoDialect_Type<"Named", "named", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
        ]> {
    let parameters = (ins
        "Type":$underlying,
        "StringAttr":$name,
        "ArrayAttr":$methods
        );
    let assemblyFormat = "`<` struct(params) `>`";
    let extraClassDefinition = [{
        ::llvm::TypeSize NamedType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
            return ::llvm::TypeSize::getFixed(8 * getTypeSize(dataLayout, params));
        }

        ::llvm::TypeSize NamedType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return ::llvm::TypeSize::getFixed(go::getDefaultTypeSize(getUnderlying(), dataLayout, params));
        }

        uint64_t NamedType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return go::getDefaultABIAlignment(getUnderlying(), dataLayout, params);
        }

        uint64_t NamedType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return go::getDefaultPreferredAlignment(getUnderlying(), dataLayout, params);
        }
    }];
}

def InterfaceType : Type<CPred<"::mlir::isa<::mlir::go::InterfaceType>($_self)">,
                        "Go Interface Type">;
def StructType : Type<CPred<"::mlir::isa<::mlir::go::GoStructType>($_self)">,
                        "Go Struct Type">;

// Any type from the given list
class AnyTypeOrNamedOf<list<Type> allowedTypes, string summary = "",
                string cppClassName = "::mlir::Type"> : Type<
    // Satisfy any of the allowed types' conditions.
    Or<[
        And<[
            CPred<"::mlir::isa<NamedType>($_self)">,
            Or<!foreach(type, allowedTypes, SubstLeaves<"$_self", "::mlir::cast<NamedType>(type).getUnderlying()", type.predicate>)>
        ]>,
        Or<!foreach(type, allowedTypes, type.predicate)>
    ]>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " or "),
        summary),
    cppClassName>;


def Go_Array : AnyTypeOrNamedOf<[ArrayType]>;
def Go_Bool : AnyTypeOrNamedOf<[BooleanType]>;
def Go_Complex : AnyTypeOrNamedOf<[Complex<F32>, Complex<F64>]>;
def Go_Chan : AnyTypeOrNamedOf<[ChanType]>;
def Go_Float : AnyTypeOrNamedOf<[F32, F64]>;
def Go_Function : AnyTypeOrNamedOf<[FunctionType]>;
def Go_Integer : AnyTypeOrNamedOf<[IntegerType]>;
def Go_Integral : AnyTypeOrNamedOf<[Go_Integer, Go_Bool]>;
def Go_String : AnyTypeOrNamedOf<[StringType]>;
def Go_Slice : AnyTypeOrNamedOf<[SliceType]>;
def Go_Pointer : AnyTypeOrNamedOf<[PointerType]>;
def Go_Map : AnyTypeOrNamedOf<[MapType]>;
def Go_Interface : AnyTypeOrNamedOf<[InterfaceType]>;
def Go_Struct : AnyTypeOrNamedOf<[StructType]>;

def Go_Numeric : AnyTypeOrNamedOf<[
    Go_Complex,
    Go_Float,
    Go_Integer
]>;

def Go_Addressable : AnyTypeOrNamedOf<[
    Go_Pointer,
    Go_Function
]>;

def Go_Aggregate : AnyTypeOrNamedOf<[
    Go_Array,
    Go_Struct
]>;

def Go_Type : AnyTypeOrNamedOf<[
    Go_Array,
    Go_Bool,
    Go_Complex,
    Go_Chan,
    Go_Float,
    Go_Function,
    Go_Integer,
    Go_Interface,
    Go_Map,
    Go_Pointer,
    Go_Slice,
    Go_String,
    Go_Struct
]>;

#endif // GO_TYPES