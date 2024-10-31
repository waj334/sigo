#ifndef GO_GoFunctionType_TD
#define GO_GoFunctionType_TD

include "Go/IR/GoDialect.td"

def GoFunctionType : GoDialect_Type<"Function", "function", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
        ]> {
    let parameters = (ins
        OptionalParameter<"::mlir::Type">:$receiver,
        ArrayRefParameter<"::mlir::Type">:$inputs,
        ArrayRefParameter<"::mlir::Type">:$results);
    let builders = [
        TypeBuilder<(ins
            "::mlir::TypeRange":$inputs,
            "::mlir::TypeRange":$results,
            CArg<"std::optional<::mlir::Type>", "std::nullopt">:$receiver)>,
    ];
    let extraClassDeclaration = [{
        ::mlir::Type getInput(size_t index) const { return this->getInputs()[index]; }
        size_t getNumInputs() const { return this->getInputs().size(); }

        ::mlir::Type getResult(size_t index) const { return this->getResults()[index]; }
        size_t getNumResults() const { return this->getResults().size(); }

        FunctionType clone(::mlir::TypeRange inputs, ::mlir::TypeRange results,
                std::optional<Type> receiver = std::nullopt)
        {
            return FunctionType::get(this->getContext(), inputs, results, receiver);
        }

        bool hasReceiver() {
            if (this->getReceiver()) {
                return true;
            }
            return false;
        }
    }];
    let extraClassDefinition = [{
        ::llvm::TypeSize FunctionType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
        }
        ::llvm::TypeSize FunctionType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSize(IndexType::get(this->getContext()));
        }
        uint64_t FunctionType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(IndexType::get(this->getContext()));
        }
        uint64_t FunctionType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
        }
    }];
    let assemblyFormat = [{
        ( `[` $receiver^ `]` )? `(` $inputs `)` `->` `(` $results `)`
    }];
}

#endif // GO_GoFunctionType_TD