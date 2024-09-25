#pragma once

#include <mlir/IR/BuiltinOps.h>

namespace mlir::go {
    class Interpreter {
    public:
        explicit Interpreter(ModuleOp *module);

        void call(mlir::FlatSymbolRefAttr symbol);

        void setCurrentOperation(Operation *op);

        Operation *currentOperation() const;

    private:
        void process();
        void advance();

        ModuleOp *mp_module;
        Operation *mp_currentOp;
    };
}
