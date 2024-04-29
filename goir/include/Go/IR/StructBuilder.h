#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>

namespace mlir::go {
    class StructBuilder {
    public:
        explicit StructBuilder(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type T);
        explicit StructBuilder(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value T);

        void Insert(uint64_t index, mlir::Value value);
        mlir::Value Value() const;

    private:
        mlir::OpBuilder& m_builder;
        mlir::Type m_structT;
        mlir::Value m_currentValue;
        mlir::Location m_loc;
    };
}
