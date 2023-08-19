#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

#include <llvm/ADT/TypeSwitch.h>

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
                    if (namedType && namedType.getName() == "func") {
                        return success();
                    }
                    return this->emitOpError() << "expected \"func\" struct type.";
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

    LogicalResult GoOp::canonicalize(GoOp op, PatternRewriter &rewriter) {
        const auto context = rewriter.getContext();
        const auto loc = op.getLoc();
        const auto calleeType = op.getCallee().getType();
        auto module = op->getParentOfType<ModuleOp>();

        const auto ptrType = PointerType::get(context, std::nullopt);
        const auto calleeOperands = op.getCalleeOperands();

        // Get the runtime function.
        auto addTaskFunc = mlir::cast<func::FuncOp>(module.lookupSymbol("runtime.addTask"));

        // Get the function value type from the runtime function signature.
        const auto funcType = *addTaskFunc.getArgumentTypes().begin();

        // Create the function value.
        Value funcValue = rewriter.create<ZeroOp>(loc, funcType);

        // Pack the call arguments (if any).
        if (!calleeOperands.empty()) {
            // Create the argument pack type.
            SmallVector<Type> elements;
            for (const auto operand: calleeOperands) {
                elements.push_back(operand.getType());
            }
            const auto argPackT = LLVM::LLVMStructType::getLiteral(context, elements);

            // Allocate memory to store the call arguments.
            const auto argsPackPtrT = PointerType::get(context, argPackT);
            Value args = rewriter.create<AllocaOp>(loc, argsPackPtrT, argPackT, Value(), UnitAttr(), StringAttr());

            // Pack the call arguments.
            for (size_t i = 0; i < calleeOperands.size(); i++) {
                const auto addrT = PointerType::get(context, elements[i]);
                const auto indices = DenseI32ArrayAttr::get(context, SmallVector{0, static_cast<int>(i)});
                Value addr = rewriter.create<GetElementPointerOp>(loc, addrT, args, argPackT, SmallVector<Value>(),
                                                                  indices);
                rewriter.create<StoreOp>(loc, calleeOperands[i], addr, UnitAttr(), UnitAttr());
            }

            // Store the arguments in the function value struct.
            funcValue = rewriter.create<InsertOp>(loc, funcType, args, 1, funcValue);
        }

        // Handle invoking the callee based on its type
        Value funcAddr;
        llvm::TypeSwitch<Type>(calleeType)
                .Case<FunctionType, PointerType>([&](auto type) {
                    // Get the address of the function.
                    funcAddr = op.getCallee();
                    if (go::isa<FunctionType>(calleeType)) {
                        // Bitcast the function value to a pointer.
                        funcAddr = rewriter.create<BitcastOp>(loc, ptrType, funcAddr);
                    }
                })
                .Case([&](InterfaceType) {
                    // TODO: Must create a call wrapper for interfaces matching this signature.
                })
                .Case([&](LLVM::LLVMStructType) {
                    // TODO: Must create a call wrapper that will call the underlying function with the argument
                    //       pointer. If the argument pointer is null, then the function pointer value from the struct
                    //       can probably be used directly without needing any wrapper call.
                });

        if (!calleeOperands.empty()) {
            // TODO: If there were call arguments, then a call wrapper method should be created that will unpack them and
            //       then call the respective function.
        } else {
            // The function can be invoked by the runtime directly, so just store the function pointer as is.
            funcValue = rewriter.create<InsertOp>(loc, funcType, funcAddr, 0, funcValue);
        }

        // Lower to runtime call.
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(op, SmallVector<Type>{}, "runtime.addTask",
                                                   SmallVector{funcValue});
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
                    if (namedType && namedType.getName() == "func") {
                        return success();
                    }
                    return this->emitOpError() << "expected \"func\" struct type.";
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
