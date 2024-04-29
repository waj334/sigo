#pragma once

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <llvm/Target/TargetMachine.h>

namespace mlir::go {

    class TypeInfoLattice : public mlir::dataflow::AbstractDenseLattice {
    public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeInfoLattice)
        using AbstractDenseLattice::AbstractDenseLattice;

        mlir::ChangeResult join(const AbstractDenseLattice& rhs) override;
        void print(raw_ostream &os) const override;

        void serialize(mlir::Type* type, const llvm::DataLayout& dataLayout);
        mlir::ChangeResult state() const;

    protected:
        //uintptr_t sizeOf(mlir::Type* type, const llvm::DataLayout& layout);

    private:
        mlir::ChangeResult m_state = mlir::ChangeResult::NoChange;
        std::vector<uint8_t> m_buffer;
        llvm::SmallDenseMap<mlir::Type, uintptr_t> m_offsetMap;
    };

    class TypeInfoAnalysis : public mlir::dataflow::DenseForwardDataFlowAnalysis<TypeInfoLattice> {
    public:
        using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

        explicit TypeInfoAnalysis(DataFlowSolver &solver, llvm::DataLayout& dataLayout);

        void visitOperation(mlir::Operation *op, const TypeInfoLattice &before, TypeInfoLattice *after) override;
        void setToEntryState(TypeInfoLattice *lattice) override {
            this->propagateIfChanged(lattice, lattice->state());
        }

    protected:
        void processOperation(mlir::Operation *op) override;

    private:
        llvm::DataLayout m_dataLayout;
    };

}