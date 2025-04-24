#ifndef GO_TYPES_SLICE_TD
#define GO_TYPES_SLICE_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// SliceType
//===----------------------------------------------------------------------===//

def SliceType : GoDialect_Type<"Slice", "slice", [
            RuntimeTypeInterface,
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                            ["getTypeSize", "getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
        ]> {
    let summary = "Slice type";
    let description = [{
        The `!go.slice` type represents a variable-sized array of element types.

        Example:
        ```mlir
        !go.slice<i32>
        ```
    }];
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = [{
       `<` $elementType `>`
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

#endif // GO_TYPES_SLICE_TD