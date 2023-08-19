
#include "Go/Analysis/TypeInfoAnalysis.h"
#include "Go/IR/GoTypes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

#include <cmath>

namespace mlir::go {

    mlir::ChangeResult TypeInfoLattice::join(const dataflow::AbstractDenseLattice &rhs) {
        return ChangeResult::Change;
    }

    void TypeInfoLattice::print(raw_ostream &os) const {

    }

    void TypeInfoLattice::serialize(mlir::Type *type, const llvm::DataLayout &dataLayout) {
        if (type->isa<mlir::go::ReflectType>()) {
            this->m_state = ChangeResult::Change;
        }
    }

    mlir::ChangeResult TypeInfoLattice::state() const {
        return this->m_state;
    }

    void TypeInfoAnalysis::visitOperation(mlir::Operation *op, const TypeInfoLattice &before, TypeInfoLattice *after) {

    }

    void TypeInfoAnalysis::processOperation(mlir::Operation *op) {
        DenseForwardDataFlowAnalysis::processOperation(op);
    }

    TypeInfoAnalysis::TypeInfoAnalysis(DataFlowSolver &solver, llvm::DataLayout &dataLayout) :
            mlir::dataflow::DenseForwardDataFlowAnalysis<TypeInfoLattice>(solver),
            m_dataLayout(dataLayout) {}
}