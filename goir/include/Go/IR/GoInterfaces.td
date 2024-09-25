#ifndef GO_INTERFACE_TypeMetadataInterface_TD
#define GO_INTERFACE_TypeMetadataInterface_TD

include "mlir/IR/OpBase.td"

include "GoTypes.td"

def TypeMetadataInterface : OpInterface<"TypeMetadataInterface"> {
    let description = [{
        This interface defines a standard API for operations to generate metadata about known types and is intended to
        extend ModuleOp.
    }];

    let cppNamespace = "::mlir::go";

    let methods = [
        InterfaceMethod<
            "This method will generate the respective metadata for the input type.",
            "::mlir::DictionaryAttr", "getTypeMetadata",
            (ins "::mlir::Type":$type)
        >,
    ];

    let extraClassDeclaration = [{
    private:
        ::mlir::NamedAttribute namedOf(::mlir::StringRef name, ::mlir::Attribute attr);
        ::mlir::TypeAttr typeAttrOf(::mlir::Type type);
        ::mlir::IntegerAttr intAttrOf(int64_t value);

        ::mlir::DenseMap<Type, ::mlir::DictionaryAttr> m_metadataCache;
        ::llvm::sys::SmartRWMutex<true> m_mutex;
    }];
}

#endif // GO_INTERFACE_TypeMetadataInterface_TD