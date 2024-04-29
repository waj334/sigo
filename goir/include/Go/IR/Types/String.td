#ifndef GO_TYPES_STRING_TD
#define GO_TYPES_STRING_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def StringType : GoDialect_Type<"String", "string", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
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
        ::llvm::TypeSize StringType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
            return this->getTypeSize(dataLayout, params) * 8;
        }

        ::llvm::TypeSize StringType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            // String type consists of a pointer and a integer (int is 32 or 64 bits)
            return dataLayout.getTypeSize(IndexType::get(getContext())) * 2;
        }

        uint64_t StringType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(IndexType::get(getContext()));
        }

        uint64_t StringType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypePreferredAlignment(IndexType::get(getContext()));
        }
    }];
}

#endif // GO_TYPES_STRING_TD
