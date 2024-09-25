#pragma once

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

namespace mlir::go {
    struct GoStructTypeStorage : public TypeStorage {
        using IdTy = StringRef;
        using FieldTy = std::tuple<mlir::StringAttr, mlir::Type, mlir::StringAttr>;
        using FieldsTy = ArrayRef<FieldTy>;
        using KeyTy = std::tuple<IdTy, FieldsTy>;

        explicit GoStructTypeStorage(FieldsTy members)
            : m_fields(std::move(members)), m_complete(true) {
            // Does nothing.
        }

        explicit GoStructTypeStorage(IdTy id)
            : m_id(std::move(id)), m_complete(false) {
            // Does nothing.
        }

        bool operator==(const KeyTy &key) const {
            const auto id = std::move(std::get<0>(key));
            const auto members = std::move(std::get<1>(key));
            if (!id.empty()) {
                return this->m_id == id;
            }
            return this->m_fields == members;
        }

        static llvm::hash_code hashKey(const KeyTy &key) {
            const auto id = std::move(std::get<0>(key));
            const auto members = std::move(std::get<1>(key));
            if (!id.empty()) {
                return llvm::hash_combine(id);
            }
            return llvm::hash_combine(members);
        }

        [[nodiscard]] KeyTy getAsKey() const {
            return {m_id, m_fields};
        }

        static GoStructTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy &&key) {
            const auto id = std::move(std::get<0>(key));
            const auto members = std::move(std::get<1>(key));
            if (!id.empty()) {
                return new(allocator.allocate<GoStructTypeStorage>()) GoStructTypeStorage(allocator.copyInto(std::move(id)));
            }

            // Create the literal struct.
            return new(allocator.allocate<GoStructTypeStorage>()) GoStructTypeStorage(
                allocator.copyInto(std::move(members)));
        }

        LogicalResult mutate(TypeStorageAllocator &allocator, FieldsTy fields) {
            if (this->m_id.empty() || this->m_complete) {
                return failure();
            }

            this->m_fields = allocator.copyInto(std::move(fields));
            this->m_complete = true;
            return success();
        }

        IdTy getId() const {
            return this->m_id;
        }

        FieldsTy getFields() const {
            return this->m_fields;
        }

        IdTy m_id;
        FieldsTy m_fields;
        bool m_complete;
    };
}
