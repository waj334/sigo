#ifndef GO_TYPES_SLICE_TD
#define GO_TYPES_SLICE_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// SliceType
//===----------------------------------------------------------------------===//

def SliceType : GoDialect_Type<"Slice", "slice", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
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
        ::llvm::TypeSize SliceType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
           return this->getTypeSize(dataLayout, params) * 8;
        }

        ::llvm::TypeSize SliceType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            // The slice type consists of a pointer and 2 integers (int type is 32 or 64 bits)
            return dataLayout.getTypeSize(IndexType::get(getContext())) * 3;
        }

        uint64_t SliceType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(IndexType::get(getContext()));
        }

        uint64_t SliceType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypePreferredAlignment(IndexType::get(getContext()));
        }
    }];
}

#endif // GO_TYPES_SLICE_TD