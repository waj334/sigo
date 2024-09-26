#ifndef GO_GoFunctionType_TD
#define GO_GoFunctionType_TD

include "Go/IR/GoDialect.td"

def GoFunctionType : GoDialect_Type<"Function", "function"> {
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
            return !!this->getReceiver();
        }
    }];
    let assemblyFormat = [{
        ( `[` $receiver^ `]` )? `(` $inputs `)` `->` `(` $results `)`
    }];
}

#endif // GO_GoFunctionType_TD