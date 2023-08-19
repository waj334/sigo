#include "Go/IR/GoOps.h"
#include "Go/Transforms/OptimizeDefersPass.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir::go {
    struct OptimizeDefersPass : public mlir::PassWrapper<OptimizeDefersPass, mlir::OperationPass<
                mlir::func::FuncOp>> {
    public:
        void runOnOperation() final {
            mlir::func::FuncOp func = getOperation();
            SmallVector<mlir::Operation*, 16> operationsToRemove;
            bool hasDefer = false;

            // Iterate over blocks in the function to find all defers and defer stacking calls
            for (Block&block: func) {
                for (mlir::Operation& op : block.getOperations()) {
                    if (mlir::isa<DeferOp>(op)) {
                        hasDefer = true;
                    }

                    if (auto callOp = mlir::dyn_cast<mlir::go::RuntimeCallOp>(op); callOp) {
                        if (auto callee = callOp.getCalleeAttr(); callee) {
                            if (callee.getValue() == "runtime.deferStartStack" || callee.getValue() == "runtime.deferRun") {
                                operationsToRemove.push_back(&op);
                            }
                        }
                    }
                }
            }

            if (!hasDefer) {
                for (auto op : operationsToRemove) {
                    op->erase();
                }
            }
        }

        StringRef getArgument() const final {
            return "go-optimize-defers-pass";
        }

        StringRef getDescription() const final {
            return "Eliminate unneccessary defers";
        }

        void getDependentDialects(DialectRegistry&registry) const override {
            registry.insert<GoDialect>();
            registry.insert<mlir::func::FuncDialect>();
            registry.insert<mlir::LLVM::LLVMDialect>();
        }
    };

    std::unique_ptr<mlir::Pass> createOptimizeDefersPass() {
        return std::make_unique<OptimizeDefersPass>();
    }
} // namespace mlir::go
