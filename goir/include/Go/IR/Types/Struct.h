#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <Go/IR/Types/StructDetail.h>

namespace mlir::go {
    class GoStructType : public mlir::Type::TypeBase<GoStructType, Type, GoStructTypeStorage,
                DataLayoutTypeInterface::Trait,
                TypeTrait::IsMutable> {
    public:
        using Base::Base;
        using FieldTy = GoStructTypeStorage::FieldTy;
        using FieldsTy = GoStructTypeStorage::FieldsTy;
        using IdTy = GoStructTypeStorage::IdTy;

        static constexpr ::llvm::StringLiteral name = "go.struct";

        static GoStructType get(MLIRContext *context, IdTy id);

        static GoStructType getBasic(MLIRContext *context, mlir::ArrayRef<Type> fieldTypes);

        static GoStructType getLiteral(MLIRContext *context, GoStructTypeStorage::FieldsTy fields);

        static constexpr ::llvm::StringLiteral getMnemonic() {
            return {"struct"};
        }

        size_t getNumFields() const;

        Type getFieldType(size_t index) const;

        SmallVector<mlir::Type> getFieldTypes() const;

        GoStructTypeStorage::FieldsTy getFields() const;

        LogicalResult setFields(FieldsTy fields);

        IdTy getId() const;

        bool isLiteral() const;

        static ::mlir::Type parse(::mlir::AsmParser &p);

        void print(::mlir::AsmPrinter &p) const;

        /// DataLayoutTypeInterface methods.
        llvm::TypeSize getTypeSize(const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) const;

        llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params) const;

        uint64_t getABIAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;

        uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const;

        uint64_t getFieldOffset(const DataLayout &dataLayout, unsigned idx) const;
    };
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::go::GoStructType)