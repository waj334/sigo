#ifndef GO_TYPES_CHAN_TD
#define GO_TYPES_CHAN_TD

include "mlir/IR/EnumAttr.td"

include "Go/IR/GoDialect.td"
include "Go/IR/GoEnums.td"

//===----------------------------------------------------------------------===//
// ChanType
//===----------------------------------------------------------------------===//

def ChanType : GoDialect_Type<"Chan", "chan", [
            RuntimeTypeInterface,
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                ["getTypeSize", "getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
        ]> {
    let summary = "Channel type";
    let description = [{
        The `!go.chan` type represents a channel.

        Example:
        ```mlir
        !go.chan<i32>
        !go.chan<i32, SendRecv>
        !go.chan<i32, SendOnly>
        !go.chan<i32, RecvOnly>
        ```
    }];
    let parameters = (ins "Type":$elementType,
                        DefaultValuedParameter<"ChanDirection", "ChanDirection::SendRecv">:$direction);
    let assemblyFormat = [{
           `<` $elementType (`,` $direction^)? `>`
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
    //let hasCustomAssemblyFormat = 1;
}

#endif // GO_TYPES_CHAN_TD