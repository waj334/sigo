#include "Go/IR/Types/Struct.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>

namespace mlir::go {
    GoStructType GoStructType::get(MLIRContext *context, IdTy id) {
        assert(!id.empty() && "id must not be empty string");
        return Base::get(context, std::move(id), FieldsTy());
    }

    GoStructType GoStructType::getBasic(MLIRContext *context, mlir::ArrayRef<Type> fieldTypes) {
        SmallVector<FieldTy> fields;
        fields.reserve(fieldTypes.size());
        for (const Type &fieldType: fieldTypes) {
            fields.emplace_back(StringAttr(), fieldType, StringAttr());
        }
        return getLiteral(context, fields);
    }

    GoStructType GoStructType::getLiteral(MLIRContext *context, FieldsTy fields) {
        return Base::get(context, IdTy(), fields);
    }

    size_t GoStructType::getNumFields() const {
        return this->getImpl()->m_fields.size();
    }

    Type GoStructType::getFieldType(size_t index) const {
        const auto member = this->getImpl()->m_fields[index];
        return std::get<1>(member);
    }

    SmallVector<mlir::Type> GoStructType::getFieldTypes() const {
        SmallVector<mlir::Type> fields;
        fields.reserve(this->getNumFields());
        for (auto field: this->getFields()) {
            fields.push_back(std::get<1>(field));
        }
        return fields;
    }

    GoStructType::FieldsTy GoStructType::getFields() const {
        return this->getImpl()->getFields();
    }

    LogicalResult GoStructType::setFields(FieldsTy members) {
        return this->mutate(members);
    }

    GoStructType::IdTy GoStructType::getId() const {
        return this->getImpl()->getId();
    }

    bool GoStructType::isLiteral() const {
        return this->getImpl()->m_id.empty();
    }

    mlir::Type GoStructType::parse(mlir::AsmParser &p) {
        std::string id;

        if (p.parseLess().failed()) {
            p.emitError(p.getCurrentLocation()) << "expected `<`";
            return {};
        }

        // Parse the id if present.
        if (auto result = p.parseOptionalKeywordOrString(&id); result.succeeded()) {
            GoStructType structType = GoStructType::get(p.getContext(), id);
            FailureOr<::mlir::AsmParser::CyclicParseReset> cyclicParse = p.tryStartCyclicParse(structType);
            if (failed(cyclicParse)) {
                // Don't parse any further.
                if (p.parseGreater().failed()) {
                    p.emitError(p.getCurrentLocation()) << "expected `>`";
                    return {};
                }
                return structType;
            }
        }

        // Parse the struct members;
        if (p.parseLBrace().failed()) {
            p.emitError(p.getCurrentLocation()) << "expected `{`";
            return {};
        }

        SmallVector<FieldTy> members;
        if (const auto result = p.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren, [&]() -> ParseResult {
            std::string name;
            std::string tags;
            Type type;

            if (p.parseKeywordOrString(&name).failed()) {
                return p.emitError(p.getCurrentLocation()) << "expected struct field name";
            }

            if (p.parseType((type)).failed()) {
                return p.emitError(p.getCurrentLocation()) << "expected struct field type";
            }

            (void) p.parseOptionalKeywordOrString(&tags);

            // Create the tuple representing the struct field member.
            const mlir::StringAttr nameAttr = mlir::StringAttr::get(type.getContext(), name);
            mlir::StringAttr tagsAttr;
            if (!tags.empty()) {
                tagsAttr = mlir::StringAttr::get(type.getContext(), tags);
            }
            members.emplace_back(nameAttr, type, tagsAttr);
            return success();
        }); result.failed()) {
            return {};
        }

        if (p.parseRBrace().failed()) {
            p.emitError(p.getCurrentLocation()) << "expected `}`";
            return {};
        }

        if (p.parseGreater().failed()) {
            p.emitError(p.getCurrentLocation()) << "expected `>`";
            return {};
        }

        if (!id.empty()) {
            auto structType = GoStructType::get(p.getContext(), id);
            if (structType.setFields(members).failed()) {
                p.emitError(p.getCurrentLocation()) << "failed to complete struct type";
            }
            return structType;
        }
        return GoStructType::getLiteral(p.getContext(), members);
    }

    void GoStructType::print(mlir::AsmPrinter &p) const {
        const auto id = this->getId();
        const auto members = this->getFields();
        auto cyclicPrint = p.tryStartCyclicPrint(*this);

        p << "<";

        // Print the id.
        if (!id.empty()) {
            p << id;
        }

        // Print the members.
        if (succeeded(cyclicPrint)) {
            p << "{";
            for (size_t i = 0; i < members.size(); i++) {
                const auto name = std::get<0>(members[i]);
                const auto type = std::get<1>(members[i]);
                const auto tags = std::get<2>(members[i]);

                p << "(";

                if (name && !name.empty()) {
                    p << name.str() << " ";
                } else {
                    p << "_ ";
                }

                p << type;

                if (tags && !tags.empty()) {
                    p << " " << tags.str();
                }

                p << ")";

                if (i != members.size() - 1) {
                    p << ", ";
                }
            }
            p << "}";
        }
        p << ">";
    }

    llvm::TypeSize GoStructType::getTypeSize(const mlir::DataLayout &dataLayout,
                                             mlir::DataLayoutEntryListRef params) const {
        assert(this->getImpl()->m_complete && "struct type must be complete");

        unsigned size = 0;
        llvm::Align alignment;
        for (size_t i = 0; i < this->getNumFields(); i++) {
            const auto memberType = this->getFieldType(i);
            const llvm::Align memberAlignment(dataLayout.getTypeABIAlignment(memberType));

            if (!llvm::isAligned(memberAlignment, size)) {
                // Account for padding of this member.
                size = llvm::alignTo(size, memberAlignment);
            }

            // Track the maximum alignment for this struct type.
            alignment = std::max(memberAlignment, alignment);

            // Add the size of the member to the overall struct size value.
            size += dataLayout.getTypeSize(memberType);
        }

        // Account for padding at the end of the struct.
        if (!llvm::isAligned(alignment, size)) {
            size = llvm::alignTo(size, alignment);
        }

        return llvm::TypeSize::getFixed(size);
    }

    llvm::TypeSize GoStructType::getTypeSizeInBits(const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
        return this->getTypeSize(dataLayout, params) * 8;
    }

    uint64_t GoStructType::getABIAlignment(const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
        assert(this->getImpl()->m_complete && "struct type must be complete");

        llvm::Align alignment;
        for (size_t i = 0; i < this->getNumFields(); i++) {
            const auto memberType = this->getFieldType(i);
            const llvm::Align memberAlignment(dataLayout.getTypeABIAlignment(memberType));
            alignment = std::max(memberAlignment, alignment);
        }
        return alignment.value();
    }

    uint64_t GoStructType::getPreferredAlignment(const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
        return this->getABIAlignment(dataLayout, params);
    }

    uint64_t GoStructType::getFieldOffset(const DataLayout &dataLayout, unsigned idx) const {
        assert(this->getImpl()->m_complete && "struct type must be complete");
        assert(idx < this->getNumFields() && "index out of bounds");

        // Fast path.
        if (idx == 0) {
            return 0;
        }

        unsigned offset = 0;
        for (size_t i = 0; i < idx; i++) {
            const auto memberType = this->getFieldType(i);
            const llvm::Align memberAlignment(dataLayout.getTypeABIAlignment(memberType));
            if (!llvm::isAligned(memberAlignment, offset)) {
                // Account for padding of this member.
                offset = llvm::alignTo(offset, memberAlignment);
            }

            // Add the size of the member to the offset value.
            offset += dataLayout.getTypeSize(memberType);
        }
        return offset;
    }
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::go::GoStructType)
