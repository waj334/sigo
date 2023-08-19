#ifndef GO_TYPES_POINTER_TD
#define GO_TYPES_POINTER_TD

include "Go/IR/GoDialect.td"

//===----------------------------------------------------------------------===//
// Pointer
//===----------------------------------------------------------------------===//

def PointerType : GoDialect_Type<"Pointer", "ptr", [
           DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
       ]> {
    let summary = "Pointer type";
    let description = [{
        The `!go.ptr` represents a pointer to a specific type.

        ```mlir
        !go.ptr         ; unsafe.Pointer
        !go.ptr<i32>
        ```
    }];
    let parameters = (ins DefaultValuedParameter<"std::optional<Type>", "{}">:$elementType);
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ::llvm::TypeSize PointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                                                              DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
        }

        ::llvm::TypeSize PointerType::getTypeSize(const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSize(IndexType::get(this->getContext()));
        }

        uint64_t PointerType::getABIAlignment(const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) const {
            return dataLayout.getTypeABIAlignment(IndexType::get(this->getContext()));
        }

        uint64_t PointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                             DataLayoutEntryListRef params) const {
            return dataLayout.getTypeSizeInBits(IndexType::get(this->getContext()));
        }
    }];
}

#endif // GO_TYPES_POINTER_TD