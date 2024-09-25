#pragma once

#include "Go/IR/GoTypes.h"
#include "Go/Util.h"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

namespace mlir::go {
    class RuntimeTypeLookUp {
    public:
        mlir::Type lookupRuntimeType(const std::string& type) const {
            auto it = this->m_typeMap.find(type);
            assert(it != this->m_typeMap.end());
            return (*it).second;
        }

        explicit RuntimeTypeLookUp(mlir::ModuleOp module);

    private:
        std::map<std::string, mlir::Type> m_typeMap;
    };

    class CoreTypeConverter : public mlir::TypeConverter, public RuntimeTypeLookUp {
    public:
        explicit CoreTypeConverter(mlir::ModuleOp module);

        ~CoreTypeConverter() override = default;

        mlir::Type convertArray(ArrayType T) const;

        template<typename TYPE>
        void addRuntimeTypeConversion() {
            static_assert(std::is_base_of<mlir::Type, TYPE>::value, "T must be derived from mlir::Type");
            this->addConversion([=](TYPE) {
                return this->convertType(this->lookupRuntimeType(TYPE::getMnemonic().str()));
            });
        }

        template<typename TYPE>
        void ignoreType() {
            static_assert(std::is_base_of<mlir::Type, TYPE>::value, "T must be derived from mlir::Type");
            this->addConversion([=](TYPE T) {
                return T;
            });
        }

    private:
        llvm::DenseMap<llvm::hash_code, mlir::Type> m_namedStructMap;
    };

    class LLVMTypeConverter : public mlir::LLVMTypeConverter, public RuntimeTypeLookUp {
    public:
        explicit LLVMTypeConverter(mlir::ModuleOp module, const mlir::LowerToLLVMOptions& options);

        mlir::Type convertPointer(PointerType T) const;

        template<typename TYPE>
        void addRuntimeTypeConversion() {
            static_assert(std::is_base_of<mlir::Type, TYPE>::value, "T must be derived from mlir::Type");
            this->addConversion([=](TYPE) {
                return this->convertType(this->lookupRuntimeType(TYPE::getMnemonic().str()));
            });
        }

    private:
        llvm::DenseMap<llvm::hash_code, mlir::Type> m_typeMap;
    };
}
