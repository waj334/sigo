#ifndef GO_INTERFACES_TD
#define GO_INTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def RangeOpInterface : OpInterface<"RangeOpInterface"> {
    let cppNamespace = "::mlir::go";
    let description = [{
        Interface providing common methods for range operation.
    }];
    let methods = [
        InterfaceMethod<"Returns the body block", "::mlir::Block*", "getBodyBlock">,
        InterfaceMethod<"Returns the exit block", "::mlir::Block*", "getExitBlock">,
        InterfaceMethod<"Verifies body block arguments", "::mlir::LogicalResult", "verifyBodyBlockArgs", (ins), [{}],
            [{
                return ::mlir::success();
            }]>
    ];
    /*
    // TODO: This verifier breaks lowering because of type conversions and operation insertions.
    let verify = [{
        ConcreteOp concreteOp = ::mlir::cast<ConcreteOp>($_op);
        if (!llvm::hasSingleElement(*concreteOp.getOperation()->getBlock()))
        {
            return concreteOp.emitOpError() << "enclosing block must contain only the range operation";
        }

        if (!concreteOp.getBodyBlock())
        {
            return concreteOp.emitOpError() << "range operation must have a body block";
        }

        if (!concreteOp.getExitBlock())
        {
            return concreteOp.emitOpError() << "range operation must have a exit block";
        }

        if (auto terminator = mlir::dyn_cast_or_null<::mlir::BranchOpInterface>(
            concreteOp.getBodyBlock()->getTerminator()))
        {
            for (size_t i = 0; i < terminator->getNumSuccessors(); i++)
            {
                if (terminator->getSuccessor(i) == concreteOp.getOperation()->getBlock())
                {
                    return mlir::success();
                }
            }
            return concreteOp.emitOpError() << "one of the body block terminator successors must point to the "
                                   "enclosing block of the range operation";
        }
        return concreteOp.verifyBodyBlockArgs();
    }];
    */
}

def RuntimeTypeModuleOpInterface : OpInterface<"RuntimeTypeModuleOpInterface"> {
    let cppNamespace = "::mlir::go";
    let description = [{
        Module interface for retrieving registered types by name
    }];
    let methods = [
        InterfaceMethod<"Returns a type by name", "::mlir::Type", "lookUpRuntimeType", (ins "::mlir::StringRef":$name), [{}],
            [{
                //auto moduleOp = ::mlir::cast<mlir::ModuleOp>($_op);
                auto moduleOp = $_op;
                assert(moduleOp.hasAttr("go.runtimeTypes") && "module MUST have the go.runtimeTypes dictionary attribute");
                auto typeMap = ::mlir::dyn_cast<mlir::DictionaryAttr>(moduleOp.getAttr("go.runtimeTypes"));
                const ::mlir::TypeAttr result = typeMap.template getAs<::mlir::TypeAttr>(name);
                if (!result)
                {
                    return ::mlir::Type();
                }
                return result.getValue();
            }]>
    ];
    let verify = [{
        return ::mlir::success(::mlir::isa<::mlir::ModuleOp>($_op));
    }];
}

def RuntimeTypeInterface : TypeInterface<"RuntimeTypeInterface"> {
    let cppNamespace = "::mlir::go";
    let description = [{
        Provides common functions for type that have runtime representations.
    }];
    let methods = [
        InterfaceMethod<"Looks up the runtime type representation for this type in the input data layout entries",
            "::mlir::Type", "getRuntimeType", (ins "::mlir::DataLayoutEntryListRef":$params), [{}],
            [{
                mlir::Type runtimeType;

                // Look up the entry for the runtime type representation of this type.
                for (auto entry : params)
                {
                    if (entry.getAbstractAttribute().getName() == "go.runtime_type_entry")
                    {
                        runtimeType = mlir::dyn_cast<mlir::TypeAttr>(entry.getValue()).getValue();
                    }
                }
                assert(runtimeType && "no runtime type representation found");
                return runtimeType;
            }]>
    ];
}

#endif // GO_INTERFACES_TD