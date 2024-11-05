#ifndef GO_TYPES_MAP_TD
#define GO_TYPES_MAP_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// MapType
//===----------------------------------------------------------------------===//

def MapType : GoDialect_Type<"Map", "map", [
            RuntimeTypeInterface,
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize", "getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
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

#endif // GO_TYPES_MAP_TD