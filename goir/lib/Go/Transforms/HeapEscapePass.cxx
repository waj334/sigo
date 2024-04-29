#include "Go/IR/GoOps.h"
#include "Go/Transforms/HeapEscapePass.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir::go {
    struct HeapEscapePass : public PassWrapper<HeapEscapePass, OperationPass<func::FuncOp> > {
        void runOnOperation() override {
            auto funcOp = getOperation();

            // Analyze all operations to find those that would cause a stack allocation to escape to the heap.
            funcOp.walk([&](Operation *op) {
                // Skip operations whose parent is NOT a function.
                if (auto parentOp = op->getParentOp(); !parentOp || !mlir::isa<func::FuncOp>(parentOp)) {
                    return;
                }

                // Analyze scenarios regarding allocation pointer usage with various operations.
                llvm::TypeSwitch<Operation *>(op)
                        .Case([](func::ReturnOp op) {
                            // A stack allocation is said to escape to the heap if its address is returned from a
                            // function directly. Analyze the returned values for stack allocations.
                            for (auto result: op->getOperands()) {
                                if (auto allocaOp = mlir::dyn_cast<AllocaOp>(result.getDefiningOp())) {
                                    // This value escapes the heap.
                                    allocaOp.setHeap(true);
                                }
                            }
                        })
                        .Case([](StoreOp op) {
                            // A stack allocation is said to escape to the heap its address is stored to some other
                            // address.
                            Value value = op.getValue();
                            if (auto definingOp = value.getDefiningOp()) {
                                if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                    // This value escapes the heap.
                                    allocaOp.setHeap(true);
                                }
                            }
                        })
                        .Case([](InsertOp op) {
                            // A stack allocation is said to escape to the heap if its address is stored to some other
                            // address. In this case the other address is a struct field.
                            Value value = op.getValue();
                            if (auto definingOp = value.getDefiningOp()) {
                                if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                    // This value escapes the heap.
                                    allocaOp.setHeap(true);
                                }
                            }
                        })
                        .Case([](func::CallOp op) {
                            // A stack allocation is said to escape to the heap its address is passed as a call argument.
                            // Analyze each call argument.
                            for (auto arg: op.getArgOperands()) {
                                if (auto definingOp = arg.getDefiningOp()) {
                                    if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                        // This value escapes the heap.
                                        allocaOp.setHeap(true);
                                    }
                                }
                            }
                        })
                        .Case([](DeferOp op) {
                            // A stack allocation is said to escape to the heap its address is passed as a call argument.
                            // Analyze each call argument.
                            for (auto arg: op.getCalleeOperands()) {
                                if (auto definingOp = arg.getDefiningOp()) {
                                    if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                        // This value escapes the heap.
                                        allocaOp.setHeap(true);
                                    }
                                }
                            }
                        })
                        .Case([](GoOp op) {
                            // A stack allocation is said to escape to the heap its address is passed as a call argument.
                            // Analyze each call argument.
                            for (auto arg: op.getCalleeOperands()) {
                                if (auto definingOp = arg.getDefiningOp()) {
                                    if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                        // This value escapes the heap.
                                        allocaOp.setHeap(true);
                                    }
                                }
                            }
                        })
                        .Case([](InterfaceCallOp op) {
                            // A stack allocation is said to escape to the heap its address is passed as a call argument.
                            // Analyze each call argument.
                            for (auto arg: op.getCalleeOperands()) {
                                if (auto definingOp = arg.getDefiningOp()) {
                                    if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                        // This value escapes the heap.
                                        allocaOp.setHeap(true);
                                    }
                                }
                            }
                        })
                        .Case([](RuntimeCallOp op) {
                            // A stack allocation is said to escape to the heap its address is passed as a call argument.
                            // Analyze each call argument.
                            for (auto arg: op.getCalleeOperands()) {
                                if (auto definingOp = arg.getDefiningOp()) {
                                    if (auto allocaOp = mlir::dyn_cast<AllocaOp>(definingOp)) {
                                        // This value escapes the heap.
                                        allocaOp.setHeap(true);
                                    }
                                }
                            }
                        });
            });
        }

        StringRef getArgument() const final {
            return "go-heap-escape-pass";
        }

        StringRef getDescription() const final {
            return "Analyzes stack allocations that escape the stack.";
        }

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<GoDialect>();
            registry.insert<func::FuncDialect>();
        }
    };

    std::unique_ptr<mlir::Pass> createHeapEscapePass() {
        return std::make_unique<HeapEscapePass>();
    }
}
