#include <Go/Transforms/TypeConverter.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Attributes.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir::go {
    std::string typeInfoSymbol(const mlir::Type &type, const std::string &prefix = "");

    mlir::LLVM::GlobalOp createTypeInfo(mlir::OpBuilder &builder, mlir::ModuleOp module, const mlir::Location &loc,
                                        const mlir::Type T);
}
