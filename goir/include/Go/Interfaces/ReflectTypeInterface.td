#ifndef GO_REFLECT_TYPE_INTERFACE_TD_
#define GO_REFLECT_TYPE_INTERFACE_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def ReflectTypeInterface : TypeInterface<"ReflectType"> {
    let cppNamespace = "::mlir::go";
    let description = [{
        This interface provides the required API for all types in order
        to support reflection.
    }];
    let methods = [
        InterfaceMethod<[{
            "Returns the size of the type in bytes."
        }],
        "uintptr_t", "sizeOf">
    ];
}

#endif // GO_REFLECT_TYPE_INTERFACE_TD_