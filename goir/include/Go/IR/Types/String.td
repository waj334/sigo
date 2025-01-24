#ifndef GO_TYPES_STRING_TD
#define GO_TYPES_STRING_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def StringType : GoDialect_Type<"String", "string", [
            RuntimeTypeInterface,
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                            ["getTypeSize", "getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
        ]> {
    let summary = "String type";
    let description = [{
        The `!go.string` type represents a character string.

        Example:
        ```mlir
        !go.string
        ```
    }];
    let extraClassDefinition = [{
        auto $cppClass::getTypeSize(const DataLayout& dataLayout, DataLayoutEntryListRef params)
          const -> ::llvm::TypeSize
        {
          return dataLayout.getTypeSize(this->getRuntimeType(params));
        }

        auto $cppClass::getTypeSizeInBits(
          const DataLayout& dataLayout,
          DataLayoutEntryListRef params) const -> ::llvm::TypeSize
        {
          return dataLayout.getTypeSizeInBits(this->getRuntimeType(params));
        }

        auto $cppClass::getABIAlignment(const DataLayout& dataLayout, DataLayoutEntryListRef params)
          const -> uint64_t
        {
          return dataLayout.getTypeABIAlignment(this->getRuntimeType(params));
        }

        auto $cppClass::getPreferredAlignment(
          const DataLayout& dataLayout,
          DataLayoutEntryListRef params) const -> uint64_t
        {
          return this->getABIAlignment(dataLayout, params);
        }
    }];
}

#endif // GO_TYPES_STRING_TD
