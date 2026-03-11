#ifndef GO_TYPES
#define GO_TYPES

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "Go/IR/GoDialect.td"
include "Go/IR/GoInterfaces.td"

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
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                            ["getTypeSize", "getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
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
            return dataLayout.getTypeSizeInBits(this->getUnderlying());
        }

        ::llvm::TypeSize NamedType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSize(this->getUnderlying());
        }

        uint64_t NamedType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(this->getUnderlying());
        }

        uint64_t NamedType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(this->getUnderlying());
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
            CPred<"::mlir::isa<::mlir::go::NamedType>($_self)">,
            Or<!foreach(type, allowedTypes, SubstLeaves<"$_self", "::mlir::cast<::mlir::go::NamedType>($_self).getUnderlying()", type.predicate>)>
        ]>,
        Or<!foreach(type, allowedTypes, type.predicate)>
    ]>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " or "),
        summary),
    cppClassName>;

class UntypedOfBasicType<string enumValue, string summary = ""> : Type<
    And<[
        CPred<"::mlir::isa<::mlir::go::UntypedType>($_self)">,
        CPred<!strconcat("::mlir::cast<::mlir::go::UntypedType>($_self).getBasicKind().getValue() == ::mlir::go::UntypedBasicKind::", enumValue)>,
    ]>,
    summary
>;

def UntypedComplex : UntypedOfBasicType<"Complex", "untyped complex">;
def UntypedBoolean : UntypedOfBasicType<"Boolean", "untyped boolean">;
def UntypedFloat : UntypedOfBasicType<"Float", "untyped float">;
def UntypedInt : UntypedOfBasicType<"Integer", "untyped integer">;
def UntypedNil : UntypedOfBasicType<"Nil", "untyped nil">;
def UntypedRune : UntypedOfBasicType<"Rune", "untyped rune">;
def UntypedString : UntypedOfBasicType<"String", "untyped string">;

def Go_Array : AnyTypeOrNamedOf<[ArrayType]>;
def Go_Bool : AnyTypeOrNamedOf<[UntypedBoolean, BooleanType]>;
def Go_Complex : AnyTypeOrNamedOf<[UntypedComplex, Complex<F32>, Complex<F64>]>;
def Go_Chan : AnyTypeOrNamedOf<[ChanType]>;
def Go_Float : AnyTypeOrNamedOf<[UntypedFloat, F32, F64]>;
def Go_Function : AnyTypeOrNamedOf<[GoFunctionType]>;
def Go_Integer : AnyTypeOrNamedOf<[UntypedInt, UntypedRune, GoIntegerType]>;
def Go_IntegerLike : AnyTypeOrNamedOf<[Go_Integer, Go_Bool]>;
def Go_String : AnyTypeOrNamedOf<[UntypedString, StringType]>;
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
    Go_Struct,
    UntypedNil
]>;

#endif // GO_TYPES