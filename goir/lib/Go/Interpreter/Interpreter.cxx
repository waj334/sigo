#include "Go/Interpreter/Interpreter.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::go {
    Interpreter::Interpreter(ModuleOp *module) : mp_module(module) {
        // TODO
    }

    void Interpreter::call(mlir::FlatSymbolRefAttr symbol) {
        // Look up the function in the module.
        if (auto op = mlir::dyn_cast<mlir::func::FuncOp>(this->mp_module->lookupSymbol(symbol)); op) {
            if (!op.getFunctionBody().empty()) {
                // Get the function's entry block
                mlir::Block& entryBlock = *op.getFunctionBody().begin();

                // Set the current operation to the beginning of the entry block.
                if (!entryBlock.empty()) {
                    this->setCurrentOperation(&*entryBlock.begin());
                }
            }
        }
    }

    void Interpreter::setCurrentOperation(Operation *op) {
        this->mp_currentOp = op;
    }

    Operation *Interpreter::currentOperation() const { return this->mp_currentOp; }

    void Interpreter::process() {
        if (this->mp_currentOp) {
            // TODO
        }
    }

    void Interpreter::advance() {
        if (this->mp_currentOp) {
            this->mp_currentOp = this->mp_currentOp->getNextNode();
        }
    }
}
