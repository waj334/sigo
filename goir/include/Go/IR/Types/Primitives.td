#ifndef GO_IntegerType_TD
#define GO_IntegerType_TD

include "Go/IR/GoDialect.td"

def BooleanType : GoDialect_Type<"Boolean", "bool"> {
}

def IntegerType : GoDialect_Type<"Integer", "integer", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
        ]> {
    let parameters = (ins
            "mlir::go::IntegerType::SignednessSemantics":$signedness,
            DefaultValuedParameter<"std::optional<unsigned>", "std::nullopt">:$width);
    let builders = [
        TypeBuilder<(ins "mlir::go::IntegerType::SignednessSemantics":$signedness,
                         CArg<"std::optional<unsigned>", "std::nullopt">:$width)>
    ];
    let extraClassDeclaration = [{
        /// Signedness semantics.
        enum SignednessSemantics : uint32_t {
          Signed,   /// Signed integer
          Unsigned, /// Unsigned integer
          Uintptr   /// Uintptr
        };

        /// Return true if this is a signed integer type.
        bool isSigned() const { return getSignedness() == Signed; }
        /// Return true if this is an unsigned integer type.
        bool isUnsigned() const { return getSignedness() == Unsigned || getSignedness() == Uintptr; }
        /// Return true if this is an pointer value.
        bool isUintptr() const { return getSignedness() == Uintptr; }
    }];

    let extraClassDefinition = [{
        ::llvm::TypeSize $cppClass::getTypeSizeInBits(const DataLayout &dataLayout,
                                                                      DataLayoutEntryListRef params) const {
            const auto width = this->getWidth();
            if (width.has_value()) {
                return ::llvm::TypeSize::getFixed(*width);
            }
            return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
        }

        ::llvm::TypeSize $cppClass::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return ::llvm::TypeSize::getFixed(this->getTypeSizeInBits(dataLayout, params).getFixedValue() / 8);
        }

        uint64_t $cppClass::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            mlir::Type T;
            const auto width = this->getWidth();
            if (width.has_value()) {
                mlir::IntegerType::get(this->getContext(), *width);
            } else {
                T = IndexType::get(this->getContext());
            }
            return dataLayout.getTypeABIAlignment(T);
        }

        uint64_t $cppClass::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return this->getABIAlignment(dataLayout, params);
        }
    }];
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif // GO_IntegerType_TD