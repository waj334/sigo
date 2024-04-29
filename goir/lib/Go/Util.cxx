#include "../../include/Go/Util.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mutex>
#include <llvm/ADT/TypeSwitch.h>

namespace mlir::go {
    static llvm::SmallDenseMap<mlir::Type, uint64_t> s_generatedTypeInfoMap = llvm::SmallDenseMap<mlir::Type,
        uint64_t>();
    static uint64_t s_typeInfoCounter = 0;
    static std::mutex s_typeInfoMutex = std::mutex();

    GoTypeId GetGoTypeId(const mlir::Type &type) {
        return TypeSwitch<Type, GoTypeId>(type)
                .Case([](IntType T) { return GoTypeId::Int; })
                .Case([](UintType T) { return GoTypeId::Uint; })
                .Case([](UintptrType T) { return GoTypeId::Uintptr; })
                .Case([](IntegerType T) {
                    if (T.getWidth() == 1) {
                        return GoTypeId::Bool;
                    }

                    switch (T.getWidth()) {
                        case 8:
                            return T.isSigned() ? GoTypeId::Int8 : GoTypeId::Uint8;
                        case 16:
                            return T.isSigned() ? GoTypeId::Int16 : GoTypeId::Uint16;
                        case 32:
                            return T.isSigned() ? GoTypeId::Int32 : GoTypeId::Uint32;
                        case 64:
                            return T.isSigned() ? GoTypeId::Int64 : GoTypeId::Uint64;
                        default:
                            return GoTypeId::Invalid;
                    }
                })
                .Case([](FloatType T) {
                    return T.getWidth() == 32 ? GoTypeId::Float32 : GoTypeId::Float64;
                })
                .Case([](ComplexType T) {
                    return mlir::cast<FloatType>(T.getElementType()).getWidth() == 32
                               ? GoTypeId::Complex64
                               : GoTypeId::Complex128;
                })
                .Case([](ArrayType T) { return GoTypeId::Array; })
                .Case([](ChanType T) { return GoTypeId::Chan; })
                .Case([](FunctionType T) { return GoTypeId::Func; })
                .Case([](InterfaceType T) { return GoTypeId::Interface; })
                .Case([](MapType T) { return GoTypeId::Map; })
                .Case([](PointerType T) {
                    return T.getElementType().has_value() ? GoTypeId::Pointer : GoTypeId::UnsafePointer;
                })
                .Case([](SliceType T) { return GoTypeId::Slice; })
                .Case([](StringType T) { return GoTypeId::String; })
                .Case([](LLVM::LLVMStructType T) { return GoTypeId::Struct; })
                .Default([](Type T) { return GoTypeId::Invalid; });
    }

    std::string typeInfoSymbol(const mlir::Type &type, const std::string &postfix) {
        const auto id = getTypeId(type);
        const std::string symbol = "type" + postfix + std::to_string(id);
        return symbol;
    }

    uint64_t getTypeId(const mlir::Type &type) {
        std::lock_guard<std::mutex> guard(s_typeInfoMutex);
        const auto it = s_generatedTypeInfoMap.find(type);
        if (it == s_generatedTypeInfoMap.end()) {
            const auto result = s_typeInfoCounter++;
            s_generatedTypeInfoMap[type] = result;
            return result;
        }
        return it->getSecond();
    }

    std::string typeStr(const mlir::Type &T) {
        std::string typeStr;
        llvm::raw_string_ostream in(typeStr);
        T.print(in);
        return typeStr;
    }

    llvm::hash_code computeMethodHash(const StringRef name, const mlir::FunctionType func, bool isInterface) {
        llvm::hash_code result = llvm::hash_value(name);
        size_t offset = 0;
        if (!isInterface) {
            // Skip the receiver for named-type methods.
            offset = 1;
        }

        // Hash the input types starting at the offset.
        for (size_t i = offset; i < func.getNumInputs(); i++) {
            std::string str;
            llvm::raw_string_ostream(str) << func.getInput(i);
            result = llvm::hash_combine(result, str);
        }

        // Hash the result types
        for (auto t: func.getResults()) {
            std::string str;
            llvm::raw_string_ostream(str) << t;
            result = llvm::hash_combine(result, str);
        }
        return result;
    }
} // namespace mlir::go
