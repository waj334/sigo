#ifndef GO_TYPES_ARRAY_TD
#define GO_TYPES_ARRAY_TD

include "Go/IR/GoDialect.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def ArrayType : GoDialect_Type<"Array", "array", [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
        ]> {
    let summary = "Array type";
    let description = [{
        The `!go.array` type represents a fixed-sized array.

        Example:
        ```mlir
        !go.array<i32[64]>
        ```
    }];
    let parameters = (ins "Type":$elementType, "unsigned":$length);
    let assemblyFormat = [{
        `<` $elementType `[` $length `]` `>`
    }];
    let extraClassDefinition = [{
        ::llvm::TypeSize ArrayType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                  DataLayoutEntryListRef params) const {
            return this->getTypeSize(dataLayout, params) * 8;
        }

        ::llvm::TypeSize ArrayType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSize(this->getElementType()) * this->getLength();
        }

        uint64_t ArrayType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
          return dataLayout.getTypeABIAlignment(getElementType());
        }

        uint64_t ArrayType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
          return dataLayout.getTypePreferredAlignment(getElementType());
        }
    }];
}

#endif // GO_TYPES_ARRAY_TD
