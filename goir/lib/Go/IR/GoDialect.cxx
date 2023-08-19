#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>

#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

//#include "Go/IR/Types/Struct.h"
#include "Go/IR/GoAttrs.h"
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoTypes.h"
#include "Go/IR/GoOps.h"

#include "Go/IR/GoOpsDialect.cpp.inc"
#include "Go/IR/GoEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES

#include "Go/IR/GoTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES

#include "Go/IR/GoAttrDefs.cpp.inc"

#include <stack>

//===----------------------------------------------------------------------===//
// Go dialect.
//===----------------------------------------------------------------------===//

namespace {
    struct GoOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
        using OpAsmDialectInterface::OpAsmDialectInterface;

        virtual AliasResult getAlias(mlir::Type type, llvm::raw_ostream &os) const override {
            return llvm::TypeSwitch<::mlir::Type, AliasResult>(type)
                    .Case<mlir::go::NamedType>([&](auto T) {
                        if (auto structType = mlir::go::dyn_cast<mlir::LLVM::LLVMStructType>(T)) {
                            if (structType.isIdentified()) {
                                // Check if this struct contains a recursive self-reference.
                                llvm::SmallDenseSet<mlir::LLVM::LLVMStructType> encountered;
                                std::stack<mlir::LLVM::LLVMStructType> stack;
                                stack.push(structType);
                                while (!stack.empty()) {
                                    auto S = stack.top();
                                    stack.pop();
                                    encountered.insert(S);

                                    for (auto &memberType: S.getBody()) {
                                        mlir::LLVM::LLVMStructType nestedStruct;
                                        if (auto memberPointer = mlir::go::dyn_cast<mlir::go::PointerType>(
                                                memberType)) {
                                            if (memberPointer.getElementType().has_value()) {
                                                if (auto ptrStruct = mlir::go::dyn_cast<mlir::LLVM::LLVMStructType>(
                                                        *memberPointer.getElementType())) {
                                                    nestedStruct = ptrStruct;
                                                }
                                            }
                                        } else if (auto memberStruct = mlir::go::dyn_cast<mlir::LLVM::LLVMStructType>(
                                                memberType)) {
                                            nestedStruct = memberStruct;
                                        }

                                        if (nestedStruct) {
                                            // Check if the struct member has the same identifier as the outer struct.
                                            if (nestedStruct.isIdentified() &&
                                                nestedStruct.getName() == structType.getName()) {
                                                // This struct is recursive.
                                                return AliasResult::NoAlias;
                                            }

                                            // Check this struct next.
                                            if (!encountered.contains(nestedStruct)) {
                                                stack.push(nestedStruct);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        os << T.getName().str();
                        return AliasResult::OverridableAlias;
                    })
                    .Default([](mlir::Type) { return AliasResult::NoAlias; });
        }
    };
}

namespace mlir::go {
    void GoDialect::initialize() {
        // Add dependent dialect
        this->getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
        this->getContext()->getOrLoadDialect<mlir::complex::ComplexDialect>();
        this->getContext()->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
        this->getContext()->getOrLoadDialect<mlir::DLTIDialect>();
        this->getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
        this->getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

        addOperations<
#define GET_OP_LIST

#include "Go/IR/GoOps.cpp.inc"

        >();

        addAttributes<
#define GET_ATTRDEF_LIST

#include "Go/IR/GoAttrDefs.cpp.inc"

        >();

        addTypes<
                mlir::go::InterfaceType,
#define GET_TYPEDEF_LIST

#include "Go/IR/GoTypes.cpp.inc"
        >();

        // Print type aliases.
        addInterface<GoOpAsmDialectInterface>();
    }

    ::mlir::Type GoDialect::parseType(::mlir::DialectAsmParser &parser) const {
        ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
        ::llvm::StringRef mnemonic;
        ::mlir::Type value;

        auto parseResult = ::generatedTypeParser(parser, &mnemonic, value);

        // NOTE: No result is returned if the type does not exist in the dialect. Attempt to parse any non-tablegen
        //       defined types.
        if (!parseResult.has_value()) {
            parseResult = ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(
                    parser, &mnemonic)
                    .Case(::mlir::go::InterfaceType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
                        value = ::mlir::go::InterfaceType::parse(parser);
                        return ::mlir::success(!!value);
                    })
                    .Default([&](llvm::StringRef, llvm::SMLoc) {
                        return std::nullopt;
                    });
        }

        if (parseResult.has_value())
            return value;

        parser.emitError(typeLoc) << "unknown  type `"
                                  << mnemonic << "` in dialect `" << getNamespace() << "`";
        return {};
    }

/// Print a type registered to this dialect.
    void GoDialect::printType(::mlir::Type type,
                              ::mlir::DialectAsmPrinter &printer) const {
        (void) ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(type)
                .Case<::mlir::go::InterfaceType>([&](auto t) {
                    printer << ::mlir::go::InterfaceType::getMnemonic();
                    t.print(printer);
                    return ::mlir::success();
                })
                .Default([&](auto t) {
                    return ::generatedTypePrinter(t, printer);
                });
    }
}

