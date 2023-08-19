#ifndef GO_TYPES_MAP_TD
#define GO_TYPES_MAP_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// MapType
//===----------------------------------------------------------------------===//

def MapType : GoDialect_Type<"Map", "map", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
        ]> {
    let summary = "Map type";
    let description = [{
        The `!go.map` type represents a collection of key-value pairs.

        Example:
        ```mlir
        !go.map<i32,string>
        ```
    }];
    let parameters = (ins "Type":$keyType, "Type":$valueType);
    let assemblyFormat = [{
       `<` $keyType `,` $valueType `>`
    }];
    let extraClassDefinition = [{
        ::llvm::TypeSize MapType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
            return this->getTypeSize(dataLayout, params) * 8;
        }
        ::llvm::TypeSize MapType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            // The slice type consists of a slice, a pointer and 2 integers (int type is 32 or 64 bits)
            return dataLayout.getTypeSize(IndexType::get(getContext())) * 7;
        }
        uint64_t MapType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(IndexType::get(getContext()));
        }
        uint64_t MapType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypePreferredAlignment(IndexType::get(getContext()));
        }
    }];
}

#endif // GO_TYPES_MAP_TD