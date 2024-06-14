#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/SmallVectorExtras.h>

namespace mlir::go {
    ::mlir::LogicalResult DeferOp::verify() {
        mlir::FunctionType Fn;
        size_t offset = 0;

        auto result = llvm::TypeSwitch<mlir::Type, LogicalResult>(baseType(this->getCallee().getType()))
                .Case<mlir::FunctionType>([&](auto T) -> LogicalResult {
                    if (this->getMethodNameAttr()) {
                        return this->emitOpError() << "a method can only be specified if the callee is an interface";
                    }

                    Fn = T;
                    return success();
                })
                .Case<InterfaceType>([&](InterfaceType T) -> LogicalResult {
                    if (!this->getMethodNameAttr()) {
                        return this->emitOpError() << "a method must be specified if the callee is an interface";
                    }

                    auto methods = T.getMethods();
                    if (auto it = methods.find(this->getMethodNameAttr().str()); it != methods.end()) {
                        Fn = it->second;

                        // Skip the receiver value.
                        offset = 1;

                        return success();
                    }
                    return this->emitOpError() << "interface has no method " << this->getMethodNameAttr();
                })
                .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType T) -> LogicalResult {
                    // The struct must be the "func" struct type.
                    auto namedType = mlir::dyn_cast<NamedType>(this->getCallee().getType());
                    if (namedType && namedType.getName() == "runtime._func") {
                        return success();
                    }
                    return this->emitOpError() << "expected \"runtime._func\" struct type, but got " << namedType;
                })
                .Default([&](auto T) { return this->emitOpError() << "unsupported callee type " << T; });

        if (failed(result)) {
            return result;
        }

        if (Fn) {
            // Assert args actually match the function operand's signature
            if (this->getCalleeOperands().size() != Fn.getNumInputs()) {
                return emitOpError() << "number of operands does not match function signature";
            }

            // Check each type to ensure that they match the function signature
            for (size_t i = offset; i < Fn.getNumInputs(); i++) {
                if (Fn.getInput(i) != this->getCalleeOperands()[i].getType()) {
                    return emitOpError() << "operand type " << i << " does not match signature";
                }
            }
        }
        return success();
    }

    ::mlir::LogicalResult GoOp::verify() {
        mlir::FunctionType Fn;
        size_t offset = 0;

        auto result = llvm::TypeSwitch<mlir::Type, LogicalResult>(baseType(this->getCallee().getType()))
                .Case<mlir::FunctionType>([&](auto T) -> LogicalResult {
                    if (this->getMethodNameAttr()) {
                        return this->emitOpError() << "a method can only be specified if the callee is an interface";
                    }

                    Fn = T;
                    return success();
                })
                .Case<InterfaceType>([&](InterfaceType T) -> LogicalResult {
                    if (!this->getMethodNameAttr()) {
                        return this->emitOpError() << "a method must be specified if the callee is an interface";
                    }

                    auto methods = T.getMethods();
                    if (auto it = methods.find(this->getMethodNameAttr().str()); it != methods.end()) {
                        Fn = it->second;

                        // Skip the receiver value.
                        offset = 1;

                        return success();
                    }
                    return this->emitOpError() << "interface has no method " << this->getMethodNameAttr();
                })
                .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType T) -> LogicalResult {
                    // The struct must be the "func" struct type.
                    auto namedType = mlir::dyn_cast<NamedType>(this->getCallee().getType());
                    if (namedType && namedType.getName() == "runtime._func") {
                        return success();
                    }
                    return this->emitOpError() << "expected \"runtime._func\" struct type.";
                })
                .Default([&](auto T) { return this->emitOpError() << "unsupported callee type " << T; });

        if (failed(result)) {
            return result;
        }

        if (Fn) {
            // Assert args actually match the function operand's signature
            if (this->getCalleeOperands().size() != Fn.getNumInputs()) {
                return emitOpError() << "number of operands does not match function signature";
            }

            // Check each type to ensure that they match the function signature
            for (size_t i = offset; i < Fn.getNumInputs(); i++) {
                if (Fn.getInput(i) != this->getCalleeOperands()[i].getType()) {
                    return emitOpError() << "operand type " << i << " does not match signature";
                }
            }
        }
        return success();
    }

    mlir::LogicalResult InterfaceCallOp::verify() {
        auto type = cast<InterfaceType>(this->getIface().getType());
        const auto methods = type.getMethods();
        const auto args = this->getCalleeOperands();
        const auto results = this->getResultTypes();

        // The callee cannot be an empty string
        if (this->getCallee().empty()) {
            return emitOpError() << "callee cannot be an empty string";
        }

        // The callee must be a method defined in the interface
        auto it = methods.find(this->getCallee().str());
        if (it == methods.cend()) {
            return emitOpError() << "callee does not exist in the specified interface type";
        }

        // The call arguments, excluding the receiver, must match
        // Fast-path
        if (args.size() != (*it).second.getNumInputs()) {
            return emitOpError() << "mismatch in number of inputs vs. method signature";
        }

        if (results.size() != (*it).second.getNumResults()) {
            return emitOpError() << "mismatch in number of results vs. method signature";
        }

        // Match the call arguments
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i].getType() != (*it).second.getInput(i)) {
                return emitOpError() << "argument type " << i << " does not match signature";
            }
        }

        // Match the call results
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i] != (*it).second.getResult(i)) {
                return emitOpError() << "result type " << i << " does not match signature";
            }
        }

        return success();
    }
} // namespace mlir::go
